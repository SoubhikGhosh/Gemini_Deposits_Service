from flask import Flask, request, jsonify, session
from flask_cors import CORS
import google.generativeai as genai
import json
import logging
import os
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass
from typing import Dict, Any, Optional
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
    
    log_file = os.path.join(log_dir, 'fd_app.log')
    handler = RotatingFileHandler(log_file, maxBytes=10000000, backupCount=5)
    handler.setFormatter(logging.Formatter(
        '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
    ))
    
    app.logger.addHandler(handler)
    app.logger.setLevel(logging.INFO)
    
    if app.debug:
        app.logger.addHandler(logging.StreamHandler())

setup_logging()

genai.configure(api_key="AIzaSyD2ArK74wBtL1ufYmpyrV2LqaOBrSi3mlU")

@dataclass
class FDSession:
    session_id: str
    conversation_history: str
    fd_info: Dict[str, Any]
    account_type: Optional[str] = None  # "FD" or "RD"
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

def get_account_details(model, prompt, conversation_history, account_type):
    system_prompt = f"""You are a banking assistant specializing in {account_type} accounts. Extract and validate account opening details from user messages.

    IMPORTANT EXTRACTION RULES:
    1. For amount:
       - Look for numbers preceded or succeeded by Rs, ₹, rupees, or just numbers in context of money
       - Extract only the numeric value
       - Example: "invest 10000 rupees" → amount should be "10000"

    2. For tenure:
       - Look for numbers followed by years, months, or days
       - Convert all to months
       - Example: "2 years" → tenure should be "24"

    3. For {account_type} Renewal Options:
       - FOR FD: the user MUST choose only ONE among Renew Principal and Interest, Renew Principal, Do not renew
       - FOR RD: the user MUST choose only ONE among Transfer to account, Convert to FD only Principal, Convert to FD Principal +Interest.
       - IMPORTANT: If the user asks something out of the scope, politely explain the available options.

    4. Nominee details are OPTIONAL

    {f'5. For {account_type} type (Regular/Tax Saver): determine the FD type based on user input'}

    Extract these fields from the user message:
    1. Principal amount
    2. Tenure (in months)
    3. {f'{account_type} type' if account_type == 'FD' else ''}
    4. {account_type} Renewal option
    5. Interest payout frequency (IF FD)
    6. Nominee name (optional)
    7. Nominee relationship (optional)

    Return ONLY a valid JSON object with these fields:
    {
        "amount": null or string,
        "tenure_months": null or string,
        "fd_type": null or string,
        "renewal_option": null or string,
        "interest_payout": null or string,
        "nominee_name": null or string,
        "nominee_relation": null or string
    }

    If a field is not found in the user message, keep it as null.
    Be precise and accurate in extraction. Do not guess or approximate values."""

    full_prompt = system_prompt + "\n\nConversation so far:\n" + conversation_history + "\n\nCurrent user message:\n" + prompt

    response = get_gemini_response(model, full_prompt)
    json_str = response.strip()
    if json_str.startswith("```json"):
        json_str = json_str[7:]
    if json_str.endswith("```"):
        json_str = json_str[:-3]

    account_info = json.loads(json_str.strip())
    return account_info

def format_confirmation(fd_info, account_type):
    friendly_fields = {
        "amount": "Principal Amount",
        "tenure_months": "Tenure",
        "fd_type": "FD Type",
        "interest_payout": "Interest Payout Frequency",
        "renewal_option": f"{account_type} Renewal Option",
        "nominee_name": "Nominee Name",
        "nominee_relation": "Nominee Relationship",
    }

    confirmation = f"Here's a summary of your {account_type} account details:\n\n"

    for field, value in fd_info.items():
        if value:
            if field == "amount":
                confirmation += f"- {friendly_fields[field]}: Rs. {value}\n"
            elif field == "tenure_months":
                years = int(value) // 12
                months = int(value) % 12
                tenure_str = []
                if years > 0:
                    tenure_str.append(f"{years} year{'s' if years > 1 else ''}")
                if months > 0:
                    tenure_str.append(f"{months} month{'s' if months > 1 else ''}")
                confirmation += f"- {friendly_fields[field]}: {' and '.join(tenure_str)}\n"
            else:
                confirmation += f"- {friendly_fields[field]}: {value}\n"

    confirmation += "\nWhat would you like to do?\n"
    confirmation += "1. Say 'confirm' to proceed with opening the account\n"
    confirmation += "2. Say 'change [field]' to modify any detail (e.g., 'change amount' or 'change tenure')\n"
    confirmation += "3. Say 'cancel' to cancel the process\n\n"
    confirmation += "Remember: Please verify all details carefully before confirming."

    return confirmation

@app.route('/api/start', methods=['POST'])
def start_account():
    try:
        data = request.get_json()
        account_type = data.get('account_type', 'FD').upper()  # Default to FD
        session_id = str(uuid.uuid4())

        if account_type not in ['FD', 'RD']:
            return jsonify({'error': 'Invalid account type'}), 400

        if account_type == 'FD':
            welcome_message = f"""Hello! I'm your {account_type} account opening assistant. I'll help you set up a Fixed Deposit account today.

To open your FD, I'll need:
1. {account_type} Type (Regular or Tax Saver)
2. Principal amount (minimum Rs. 1000 for Regular, Rs. 5000 for Tax Saver)
3. Tenure (minimum 3 months)
4. Interest payout frequency (monthly/quarterly/at maturity)
5. Renewal options: Renew Principal and Interest, Renew Principal, Do not renew.
6. Nominee details (optional)

You can provide these details all at once or one by one. How would you like to proceed?

Remember: Higher tenures typically offer better interest rates."""
        else:
            welcome_message = f"""Hello! I'm your {account_type} account opening assistant. I'll help you set up a Recurring Deposit account today.

To open your RD, I'll need:
1. Principal amount (minimum Rs. 5000)
2. Tenure (minimum 3 months)
3. Renewal options: Transfer to account, Convert to FD only Principal, Convert to FD Principal +Interest.
4. Nominee details (optional)

You can provide these details all at once or one by one. How would you like to proceed?"""

        fd_session = FDSession(
            session_id=session_id,
            conversation_history=f"System: {welcome_message}\n",
            fd_info={
                "amount": None,
                "tenure_months": None,
                "fd_type": None if account_type == 'FD' else None,
                "interest_payout": None if account_type == 'RD' else None,
                "renewal_option": None,
                "nominee_name": None,
                "nominee_relation": None
            },
            account_type=account_type,
            created_at=datetime.utcnow(),
            last_updated=datetime.utcnow()
        )

        session['fd'] = {
            'session_id': fd_session.session_id,
            'conversation_history': fd_session.conversation_history,
            'fd_info': fd_session.fd_info,
            'account_type': fd_session.account_type,
            'created_at': fd_session.created_at.isoformat(),
            'last_updated': fd_session.last_updated.isoformat()
        }

        app.logger.info(f"Started new {account_type} session: {session_id}")

        return jsonify({
            'message': f'{account_type} session started',
            'session_id': session_id,
            'next_prompt': welcome_message
        })

    except Exception as e:
        app.logger.error(f"Error starting {account_type} session: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/process', methods=['POST'])
def process_account():
    try:
        data = request.get_json()
        user_input = data.get('message', '').strip()
        session_id = data.get('session_id')

        if not user_input:
            return jsonify({'error': 'Message is required'}), 400

        if not session_id:
            return jsonify({'error': 'Session ID is required'}), 400

        fd_data = session.get('fd')
        if not fd_data or fd_data['session_id'] != session_id:
            return jsonify({'error': 'No active session'}), 404

        fd_data['conversation_history'] += f"User: {user_input}\n"
        account_type = fd_data['account_type']
        user_input_lower = user_input.lower()

        # Handle cancellation
        if user_input_lower in ["exit", "quit", "cancel", "stop"]:
            session.pop('fd', None)
            return jsonify({
                'message': f'{account_type} account opening cancelled. Thank you for considering our services!',
                'fd_info': fd_data['fd_info']
            })

        # Handle confirmation
        if user_input_lower == "confirm":
            # Validate minimum amount
            min_amount = 1000 if account_type == 'FD' and fd_data['fd_info'].get('fd_type') != 'Tax Saver' else 5000
            if fd_data['fd_info']['amount'] and float(fd_data['fd_info']['amount']) < min_amount:
                return jsonify({
                    'error': 'Invalid amount',
                    'next_prompt': f"The minimum amount is Rs. {min_amount}. Please provide a higher amount."
                }), 400

            # Validate minimum tenure
            if fd_data['fd_info']['tenure_months'] and int(fd_data['fd_info']['tenure_months']) < 3:
                return jsonify({
                    'error': 'Invalid tenure',
                    'next_prompt': "The minimum tenure is 3 months. Please provide a longer tenure."
                }), 400

            required_fields = ["amount", "tenure_months", "renewal_option"]
            if account_type == 'FD':
                required_fields.append("fd_type")
                required_fields.append("interest_payout")

            missing_fields = [field for field in required_fields if not fd_data['fd_info'].get(field)]

            if missing_fields:
                return jsonify({
                    'error': f'Incomplete {account_type} details',
                    'missing_fields': missing_fields,
                    'next_prompt': "Please provide the following missing information: " +
                                 ", ".join(missing_fields)
                }), 400
            else:
                session.pop('fd', None)
                return jsonify({
                    'message': f'{account_type} account opening initiated successfully! You will receive confirmation details shortly.',
                    'fd_details': fd_data['fd_info']
                })

        # Handle field changes
        if user_input_lower.startswith("change "):
            field = user_input_lower[7:].strip()
            field_mappings = {
                "amount": "amount",
                "tenure": "tenure_months",
                "fd_type": "fd_type",
                "interest": "interest_payout",
                "payout": "interest_payout",
                "renewal": "renewal_option",
                "nominee": "nominee_name",
                "relationship": "nominee_relation"
            }

            if field in field_mappings:
                fd_data['fd_info'][field_mappings[field]] = None
                if field in ["nominee", "relationship"]:
                    fd_data['fd_info']["nominee_name"] = None
                    fd_data['fd_info']["nominee_relation"] = None
                return jsonify({
                    'fd_info': fd_data['fd_info'],
                    'next_prompt': f"Please provide the new {field}:"
                })

        # Process normal input
        model = get_gemini_model()
        updated_info = get_account_details(model, user_input, fd_data['conversation_history'], account_type)

        # Update account info with new values
        for key, value in updated_info.items():
            if value:
                fd_data['fd_info'][key] = value

        fd_data['last_updated'] = datetime.utcnow().isoformat()
        session['fd'] = fd_data

        # Check if all required fields are filled
        required_fields = ["amount", "tenure_months", "renewal_option"]
        if account_type == 'FD':
            required_fields.append("fd_type")
            required_fields.append("interest_payout")

        missing_fields = [field for field in required_fields if not fd_data['fd_info'].get(field)]
        if not missing_fields:
            next_prompt = format_confirmation(fd_data['fd_info'], account_type)
        else:
            next_prompt = get_gemini_response(model,
                "Based on this account info: " + json.dumps(fd_data['fd_info']) +
                f"\nPolitely ask the user for the missing required information for {account_type} account. Be specific about what's needed.")

        response = {
            'fd_info': fd_data['fd_info'],
            'next_prompt': next_prompt
        }

        app.logger.info(f"Processed {account_type} message for session {session_id}")
        return jsonify(response)

    except Exception as e:
        app.logger.error(f"Error processing {account_type} request: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/complete', methods=['POST'])
def complete_account():
    try:
        data = request.get_json()
        session_id = data.get('session_id')

        if not session_id:
            return jsonify({'error': 'Session ID is required'}), 400

        fd_data = session.get('fd')
        if not fd_data or fd_data['session_id'] != session_id:
            return jsonify({'error': 'No active session'}), 404

        account_type = fd_data['account_type']

        required_fields = ["amount", "tenure_months", "renewal_option"]
        if account_type == 'FD':
            required_fields.append("fd_type")
            required_fields.append("interest_payout")
        missing_fields = [field for field in required_fields if not fd_data['fd_info'].get(field)]

        if missing_fields:
            session.pop('fd', None)
            return jsonify({
                'message': f'{account_type} account opening process aborted',
                'missing_fields': missing_fields,
                'fd_details': fd_data['fd_info']
            }), 201

        session.pop('fd', None)

        app.logger.info(f"Completed {account_type} account opening for session {session_id}")

        return jsonify({
            'message': f'{account_type} account opening completed successfully',
            'fd_details': fd_data['fd_info']
        })

    except Exception as e:
        app.logger.error(f"Error completing {account_type} account opening: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.utcnow().isoformat()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5001)))