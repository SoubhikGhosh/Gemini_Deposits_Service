from flask import Flask, request, jsonify, session
from flask_cors import CORS
import google.generativeai as genai
import json
import logging
import os
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass
from typing import Dict, Any
import uuid

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-change-in-production')
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=1)

CORS(app, resources={r"/*": {"origins": "*", "supports_credentials": True}})

def setup_logging():
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, 'deposit_app.log')
    handler = RotatingFileHandler(log_file, maxBytes=10000000, backupCount=5)
    handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s'))
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.INFO)
    if app.debug:
        app.logger.addHandler(logging.StreamHandler())

setup_logging()
genai.configure(api_key="YOUR_API_KEY_HERE")

@dataclass
class DepositSession:
    session_id: str
    conversation_history: str
    deposit_info: Dict[str, Any]
    created_at: datetime
    last_updated: datetime

def get_gemini_model():
    return genai.GenerativeModel(
        "gemini-pro",
        safety_settings=[
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
        ]
    )

def get_gemini_response(model, prompt):
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        app.logger.error(f"Gemini API error: {str(e)}")
        return '{}'

def get_deposit_details(model, prompt, conversation_history):
    system_prompt = """You are a precise and accurate banking assistant specializing in deposits. Extract and validate deposit account opening details from user messages.

    IMPORTANT RULES:
    1. First determine deposit type and sub-type:
       - Look for mentions of "fixed deposit", "FD", "recurring deposit", "RD"
       - For FD: Determine if Regular FD or Tax Saver FD
       - Set deposit_type as "FD" or "RD"
       - Set fd_type as "REGULAR" or "TAX_SAVER" for FDs
       
    2. For amount validation:
       - FD Regular: Minimum Rs. 1,000
       - FD Tax Saver: Minimum Rs. 5,000
       - RD: Minimum Rs. 5,000 monthly installment
       - Extract only numeric value
       - Example: "invest 10000 rupees" → amount should be "10000"
       
    3. For tenure:
       - Look for numbers followed by years, months, or days
       - Convert all to months
       - Minimum 3 months for all deposit types
       - Tax Saver FD fixed at 60 months (5 years)
       - Example: "2 years" → tenure should be "24"
       
    4. For interest payout (FD only):
       - Valid options: MONTHLY, QUARTERLY, AT_MATURITY
       - Default to AT_MATURITY if not specified
       
    5. For renewal options:
       FD Options (MANDATORY):
       - RENEW_PRINCIPAL_AND_INTEREST
       - RENEW_PRINCIPAL_ONLY
       - DO_NOT_RENEW
       
       RD Options (MANDATORY):
       - TRANSFER_TO_ACCOUNT
       - CONVERT_TO_FD_PRINCIPAL
       - CONVERT_TO_FD_PRINCIPAL_AND_INTEREST
       
    6. For nominee details (OPTIONAL):
       - Look for name and relationship
       - Properly capitalize names
       - Example: "nominee is my son john" → nominee_name: "John", nominee_relation: "Son"

    Return ONLY a valid JSON object with these fields:
    {
        "deposit_type": "FD" or "RD" or null,
        "fd_type": "REGULAR" or "TAX_SAVER" or null,
        "amount": null or string,
        "tenure_months": null or string,
        "interest_payout": "MONTHLY" or "QUARTERLY" or "AT_MATURITY" or null,
        "renewal_option": null or string,
        "nominee_name": null or string,
        "nominee_relation": null or string
    }

    Validate:
    - FD Regular: amount >= 1000
    - FD Tax Saver: amount >= 5000, tenure = 60 months
    - RD: amount >= 5000
    - All: tenure >= 3 months
    - Renewal option must match deposit type
    - Interest payout only for FD
    
    If any validation fails, set the corresponding field to null."""
    
    full_prompt = system_prompt + "\n\nConversation so far:\n" + conversation_history + "\n\nCurrent user message:\n" + prompt
    response = get_gemini_response(model, full_prompt)
    json_str = response.strip()
    if json_str.startswith("```json"):
        json_str = json_str[7:]
    if json_str.endswith("```"):
        json_str = json_str[:-3]
    return json.loads(json_str.strip())

def create_new_session():
    session_id = str(uuid.uuid4())
    welcome_message = """Hello! I'm your deposit account opening assistant. I can help you open either a Fixed Deposit (FD) or Recurring Deposit (RD) account.

1. Fixed Deposit (FD):
   - Regular FD (Min. Rs. 1,000)
   - Tax Saver FD (Min. Rs. 5,000, 5-year lock-in)
   - Interest payout: Monthly/Quarterly/At Maturity
   - Renewal options: Renew Principal & Interest/Principal Only/No Renewal

2. Recurring Deposit (RD):
   - Monthly installments (Min. Rs. 5,000)
   - Interest paid at maturity
   - Maturity options: Transfer to Account/Convert to FD

Optional: You can add a nominee to your deposit account.

Please specify your choice or ask any questions."""

    deposit_session = DepositSession(
        session_id=session_id,
        conversation_history=f"System: {welcome_message}\n",
        deposit_info={
            "deposit_type": None,
            "fd_type": None,
            "amount": None,
            "tenure_months": None,
            "interest_payout": None,
            "renewal_option": None,
            "nominee_name": None,
            "nominee_relation": None
        },
        created_at=datetime.utcnow(),
        last_updated=datetime.utcnow()
    )
    
    session['deposit'] = {
        'session_id': deposit_session.session_id,
        'conversation_history': deposit_session.conversation_history,
        'deposit_info': deposit_session.deposit_info,
        'created_at': deposit_session.created_at.isoformat(),
        'last_updated': deposit_session.last_updated.isoformat()
    }
    
    return session_id, welcome_message

def validate_deposit_info(deposit_info):
    errors = []
    
    if deposit_info['deposit_type'] == 'FD':
        if not deposit_info['fd_type']:
            errors.append("Please specify FD type (Regular or Tax Saver)")
        elif deposit_info['fd_type'] == 'REGULAR':
            if deposit_info['amount'] and float(deposit_info['amount']) < 1000:
                errors.append("Regular FD minimum amount is Rs. 1,000")
        elif deposit_info['fd_type'] == 'TAX_SAVER':
            if deposit_info['amount'] and float(deposit_info['amount']) < 5000:
                errors.append("Tax Saver FD minimum amount is Rs. 5,000")
            if deposit_info['tenure_months'] and int(deposit_info['tenure_months']) != 60:
                errors.append("Tax Saver FD tenure must be 5 years (60 months)")
    
    elif deposit_info['deposit_type'] == 'RD':
        if deposit_info['amount'] and float(deposit_info['amount']) < 5000:
            errors.append("RD minimum monthly installment is Rs. 5,000")
    
    if deposit_info['tenure_months'] and int(deposit_info['tenure_months']) < 3:
        errors.append("Minimum tenure is 3 months")
    
    return errors

def format_confirmation(deposit_info):
    friendly_fields = {
        "deposit_type": "Deposit Type",
        "fd_type": "FD Type",
        "amount": "Amount",
        "tenure_months": "Tenure",
        "interest_payout": "Interest Payout",
        "renewal_option": "Renewal Option",
        "nominee_name": "Nominee Name",
        "nominee_relation": "Nominee Relationship"
    }
    
    confirmation = f"Here's a summary of your {deposit_info['deposit_type']}"
    if deposit_info['fd_type']:
        confirmation += f" ({deposit_info['fd_type']})"
    confirmation += " account details:\n\n"
    
    for field, value in deposit_info.items():
        if value:
            if field == "amount":
                amount_prefix = "Monthly Installment" if deposit_info['deposit_type'] == "RD" else "Principal Amount"
                confirmation += f"- {amount_prefix}: Rs. {value}\n"
            elif field == "tenure_months":
                years = int(value) // 12
                months = int(value) % 12
                tenure_str = []
                if years > 0:
                    tenure_str.append(f"{years} year{'s' if years > 1 else ''}")
                if months > 0:
                    tenure_str.append(f"{months} month{'s' if months > 1 else ''}")
                confirmation += f"- {friendly_fields[field]}: {' and '.join(tenure_str)}\n"
            elif field == "interest_payout" and deposit_info['deposit_type'] == "RD":
                continue
            else:
                confirmation += f"- {friendly_fields[field]}: {value}\n"
    
    confirmation += "\nWhat would you like to do?\n"
    confirmation += "1. Say 'confirm' to proceed\n"
    confirmation += "2. Say 'change [field]' to modify (e.g., 'change amount')\n"
    confirmation += "3. Say 'cancel' to start over\n"
    confirmation += "4. Say 'exit' to quit\n"
    
    return confirmation

@app.route('/api/start', methods=['POST'])
def start_deposit():
    try:
        session_id, welcome_message = create_new_session()
        app.logger.info(f"Started new deposit session: {session_id}")
        return jsonify({
            'message': 'Deposit session started',
            'session_id': session_id,
            'next_prompt': welcome_message,
            'proceed_to_pay': False
        })
    except Exception as e:
        app.logger.error(f"Error starting deposit session: {str(e)}")
        return jsonify({'error': 'Internal server error', 'proceed_to_pay': False}), 500

@app.route('/api/process', methods=['POST'])
def process_deposit():
    try:
        data = request.get_json()
        user_input = data.get('message', '').strip()
        session_id = data.get('session_id')
        
        if not user_input or not session_id:
            return jsonify({
                'error': 'Message and session ID are required',
                'proceed_to_pay': False
            }), 400
        
        deposit_data = session.get('deposit')
        if not deposit_data or deposit_data['session_id'] != session_id:
            return jsonify({
                'error': 'Invalid session',
                'proceed_to_pay': False
            }), 404
        
        deposit_data['conversation_history'] += f"User: {user_input}\n"
        user_input_lower = user_input.lower()
        
        if user_input_lower in ["exit", "quit", "cancel", "stop"]:
            session.pop('deposit', None)
            new_session_id, welcome_message = create_new_session()
            return jsonify({
                'message': 'Session cancelled. Starting new session.',
                'session_id': new_session_id,
                'next_prompt': welcome_message,
                'proceed_to_pay': False
            })
        
        if user_input_lower == "confirm":
            errors = validate_deposit_info(deposit_data['deposit_info'])
            if errors:
                return jsonify({
                    'error': 'Validation failed',
                    'next_prompt': "Please correct the following:\n" + "\n".join(errors),
                    'proceed_to_pay': False
                }), 400
            
            required_fields = ['deposit_type', 'amount', 'tenure_months', 'renewal_option']
            if deposit_data['deposit_info']['deposit_type'] == 'FD':
                required_fields.extend(['fd_type', 'interest_payout'])
            
            missing_fields = [field for field in required_fields 
                            if not deposit_data['deposit_info'].get(field)]
            
            if missing_fields:
                return jsonify({
                    'error': 'Incomplete details',
                    'next_prompt': "Please provide: " + ", ".join(missing_fields),
                    'proceed_to_pay': False
                }), 400
            
            session.pop('deposit', None)
            return jsonify({
                'message': 'Deposit account opening initiated!',
                'deposit_details': deposit_data['deposit_info'],
                'proceed_to_pay': True
            })
        
        if user_input_lower.startswith("change "):
            field = user_input_lower[7:].strip()
            field_mappings = {
                "type": ["deposit_type", "fd_type"],
                "amount": ["amount"],
                "tenure": ["tenure_months"],
                "interest": ["interest_payout"],
                "payout": ["interest_payout"],
                "renewal": ["renewal_option"],
                "nominee": ["nominee_name", "nominee_relation"]
            }
            
            if field in field_mappings:
                for field_key in field_mappings[field]:
                    deposit_data['deposit_info'][field_key] = None
                return jsonify({
                    'deposit_info': deposit_data['deposit_info'],
                    'next_prompt': f"Please provide new {field}:",
                    'proceed_to_pay': False
                })
        
        model = get_gemini_model()
        updated_info = get_deposit_details(model, user_input, deposit_data['conversation_history'])
        
        # Update deposit info with new values
        for key, value in updated_info.items():
            if value:
                deposit_data['deposit_info'][key] = value
        
        deposit_data['last_updated'] = datetime.utcnow().isoformat()
        session['deposit'] = deposit_data
        
        # Validate current info
        errors = validate_deposit_info(deposit_data['deposit_info'])
        if errors:
            next_prompt = "Please correct the following:\n" + "\n".join(errors)
        else:
            # Check for missing required fields
            required_fields = ['deposit_type', 'amount', 'tenure_months', 'renewal_option']
            if deposit_data['deposit_info']['deposit_type'] == 'FD':
                required_fields.extend(['fd_type', 'interest_payout'])
            
            missing_fields = [field for field in required_fields 
                            if not deposit_data['deposit_info'].get(field)]
            
            if not missing_fields:
                next_prompt = format_confirmation(deposit_data['deposit_info'])
            else:
                next_prompt = get_gemini_response(model, 
                    "Based on this deposit info: " + json.dumps(deposit_data['deposit_info']) +
                    "\nPolitely ask the user for the missing required information: " + 
                    ", ".join(missing_fields))
        
        response = {
            'deposit_info': deposit_data['deposit_info'],
            'next_prompt': next_prompt,
            'proceed_to_pay': False
        }
        
        app.logger.info(f"Processed deposit message for session {session_id}")
        return jsonify(response)
    
    except Exception as e:
        app.logger.error(f"Error processing deposit message: {str(e)}")
        return jsonify({'error': 'Internal server error', 'proceed_to_pay': False}), 500

@app.route('/api/validate', methods=['POST'])
def validate_deposit():
    try:
        data = request.get_json()
        deposit_info = data.get('deposit_info', {})
        
        errors = validate_deposit_info(deposit_info)
        
        required_fields = ['deposit_type', 'amount', 'tenure_months', 'renewal_option']
        if deposit_info.get('deposit_type') == 'FD':
            required_fields.extend(['fd_type', 'interest_payout'])
        
        missing_fields = [field for field in required_fields if not deposit_info.get(field)]
        
        if errors or missing_fields:
            return jsonify({
                'valid': False,
                'errors': errors,
                'missing_fields': missing_fields
            })
        
        return jsonify({
            'valid': True
        })
    
    except Exception as e:
        app.logger.error(f"Error validating deposit: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/complete', methods=['POST'])
def complete_deposit():
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        payment_reference = data.get('payment_reference')  # Optional payment confirmation
        
        if not session_id:
            return jsonify({
                'error': 'Session ID is required',
                'status': 'FAILED'
            }), 400
        
        # Get session data
        deposit_data = session.get('deposit')
        if not deposit_data or deposit_data['session_id'] != session_id:
            return jsonify({
                'error': 'Invalid or expired session',
                'status': 'FAILED'
            }), 404
        
        # Final validation of deposit details
        deposit_info = deposit_data['deposit_info']
        errors = validate_deposit_info(deposit_info)
        
        required_fields = ['deposit_type', 'amount', 'tenure_months', 'renewal_option']
        if deposit_info.get('deposit_type') == 'FD':
            required_fields.extend(['fd_type', 'interest_payout'])
        
        missing_fields = [field for field in required_fields if not deposit_info.get(field)]
        
        if errors or missing_fields:
            return jsonify({
                'error': 'Invalid deposit details',
                'status': 'FAILED',
                'validation_errors': errors,
                'missing_fields': missing_fields
            }), 400
        
        # Create completion record
        completion_record = {
            'session_id': session_id,
            'deposit_info': deposit_info,
            'payment_reference': payment_reference,
            'status': 'COMPLETED',
            'completed_at': datetime.utcnow().isoformat(),
            'conversation_history': deposit_data['conversation_history']
        }
        
        # Here you would typically:
        # 1. Save completion record to database
        # 2. Trigger any necessary notifications
        # 3. Initialize account opening process
        # 4. Generate any required documents
        
        # Clear session
        session.pop('deposit', None)
        
        return jsonify({
            'status': 'COMPLETED',
            'message': 'Deposit account opening completed successfully',
            'reference_number': str(uuid.uuid4()),  # Generate unique reference number
            'completion_details': {
                'deposit_type': deposit_info['deposit_type'],
                'amount': deposit_info['amount'],
                'tenure_months': deposit_info['tenure_months'],
                'completed_at': completion_record['completed_at']
            }
        })
    
    except Exception as e:
        app.logger.error(f"Error completing deposit session: {str(e)}")
        # Attempt to cleanup session in case of error
        session.pop('deposit', None)
        return jsonify({
            'error': 'Internal server error',
            'status': 'FAILED'
        }), 500

@app.route('/api/rates', methods=['GET'])
def get_deposit_rates():
    try:
        # You would typically fetch these from a database or external service
        rates = {
            'FD': {
                'REGULAR': {
                    '3_6_months': 5.50,
                    '6_12_months': 6.00,
                    '1_2_years': 6.50,
                    '2_3_years': 6.75,
                    'above_3_years': 7.00
                },
                'TAX_SAVER': {
                    '5_years': 7.25
                }
            },
            'RD': {
                '6_months': 5.75,
                '12_months': 6.25,
                '24_months': 6.50,
                '36_months': 6.75,
                '60_months': 7.00
            }
        }
        
        return jsonify({
            'rates': rates,
            'last_updated': datetime.utcnow().isoformat()
        })
    
    except Exception as e:
        app.logger.error(f"Error fetching deposit rates: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True)