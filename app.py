import os
import random
import string
import re
import json
import io
import base64
import hashlib # <--- NEW: For PayHere Signature
import time
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash, g, make_response
from dotenv import load_dotenv
from datetime import datetime, timezone
from functools import wraps
import requests

# --- AI Imports ---
import google.generativeai as genai
import stability_sdk.client as StabilityClient
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

# --- Firebase & Cloudinary ---
import firebase_admin
from firebase_admin import credentials, firestore, auth
import cloudinary
import cloudinary.uploader
import cloudinary.api

# --- Load Environment Variables ---
load_dotenv() 

# --- Firebase Init (Vercel Compatible) ---
service_account_json_string = os.environ.get('FIREBASE_SERVICE_ACCOUNT_JSON')

try:
    if service_account_json_string:
        service_account_info = json.loads(service_account_json_string)
        cred = credentials.Certificate(service_account_info)
    else:
        cred = credentials.Certificate('serviceAccountKey.json')
    
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
except Exception as e:
    print(f"Firebase Init Error: {e}")

db = firestore.client()

# --- Flask Config ---
base_dir = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__, template_folder=os.path.join(base_dir, 'templates'))
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'default_secret')

# --- API Configurations ---
api_key = os.environ.get('GEMINI_API_KEY')
genai.configure(api_key=api_key) 
model_flash = genai.GenerativeModel('gemini-2.5-flash') 

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
PRO_MODEL_NAME = "cognitivecomputations/dolphin-mistral-24b-venice-edition:free"
openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")

stability_api_key = os.environ.get("STABILITY_API_KEY")
if stability_api_key:
    stability_api = StabilityClient.StabilityInference(
        key=stability_api_key, verbose=True, engine="stable-diffusion-xl-1024-v1-0"
    )

cloudinary.config( 
    cloud_name = os.environ.get("CLOUDINARY_CLOUD_NAME"), 
    api_key = os.environ.get("CLOUDINARY_API_KEY"), 
    api_secret = os.environ.get("CLOUDINARY_API_SECRET") 
)

# ==========================================
# === NEW: PAYHERE CONFIGURATION ===
# ==========================================
PAYHERE_MERCHANT_ID = os.environ.get('PAYHERE_MERCHANT_ID')
PAYHERE_MERCHANT_SECRET = os.environ.get('PAYHERE_MERCHANT_SECRET')
PAYHERE_MODE = os.environ.get('PAYHERE_MODE', 'sandbox') # 'sandbox' or 'live'
DOMAIN_URL = os.environ.get('DOMAIN_URL', 'http://127.0.0.1:5000') # Vercel URL in prod

PAYHERE_URL = "https://sandbox.payhere.lk/pay/checkout" if PAYHERE_MODE == 'sandbox' else "https://www.payhere.lk/pay/checkout"

def generate_payhere_hash(merchant_id, order_id, amount, currency, merchant_secret):
    # PayHere Hash: upper(md5(merchant_id + order_id + amount + currency + upper(md5(merchant_secret))))
    # Amount must be formatted to 2 decimal places
    amount_formatted = "{:.2f}".format(float(amount))
    hashed_secret = hashlib.md5(merchant_secret.encode('utf-8')).hexdigest().upper()
    hash_string = f"{merchant_id}{order_id}{amount_formatted}{currency}{hashed_secret}"
    return hashlib.md5(hash_string.encode('utf-8')).hexdigest().upper()

# ==========================================
# === NEW: WALLET & DAILY COIN LOGIC ===
# ==========================================
def get_or_create_wallet(uid):
    """
    Fetches user wallet.
    Checks if 'last_daily_reset' was yesterday. 
    If so, adds 20 free coins.
    """
    user_ref = db.collection('users').document(uid)
    doc = user_ref.get()
    
    now = datetime.utcnow().replace(tzinfo=timezone.utc)
    today_str = now.strftime('%Y-%m-%d')
    
    if not doc.exists:
        # New user: Give initial 20 coins
        data = {
            'coins': 20,
            'last_daily_reset': today_str,
            'email': g.user.get('email')
        }
        user_ref.set(data)
        return data

    data = doc.to_dict()
    last_reset = data.get('last_daily_reset')
    
    # Daily Reset Logic
    if last_reset != today_str:
        user_ref.update({
            'coins': firestore.Increment(20), # Add 20 coins daily
            'last_daily_reset': today_str
        })
        data['coins'] = data.get('coins', 0) + 20
        data['last_daily_reset'] = today_str
        
    return data

# === Authentication Decorator ===
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        id_token = request.headers.get('Authorization')
        if not id_token:
            id_token = session.get('id_token')
            if not id_token:
                return redirect(url_for('login'))

        if id_token.startswith('Bearer '):
            id_token = id_token.split(' ')[1]
            
        try:
            decoded_token = auth.verify_id_token(id_token)
            g.user = decoded_token
            g.user_uid = decoded_token['uid']
        except Exception as e:
            print(f"Token verification error: {e}")
            return redirect(url_for('login'))
        
        return f(*args, **kwargs)
    return decorated_function

# === Context Processor: Make Coins Available to Navbar ===
@app.context_processor
def inject_coins():
    if hasattr(g, 'user_uid'):
        # This call handles the daily reset logic automatically!
        wallet = get_or_create_wallet(g.user_uid)
        return dict(user_coins=wallet.get('coins', 0))
    return dict(user_coins=0)

# --- Helper Functions ---
def generate_room_code(length=4):
    while True:
        code = ''.join(random.choices(string.digits, k=length))
        room_ref = db.collection('rooms').document(code)
        if not room_ref.get().exists:
            return code

def build_gemini_prompt(room_data, user_nickname, message_text):
    # ... (Keep your existing prompt builder logic unchanged) ...
    prompt_lines = [
        f"You are {room_data['bot_name']}. Your personality is: {room_data['bot_personality']}.",
        f"Your appearance is: {room_data.get('bot_appearance', 'not specified')}",
        "You are in a role-playing game where users are trying to win your affection.",
        "\nCurrent affection levels:"
    ]
    if not room_data.get('users'):
        prompt_lines.append("No one is in the room yet.")
    else:
        for user_id, data in room_data['users'].items():
            prompt_lines.append(f"- {data['nickname']}: {data['score']}%")

    prompt_lines.append(f"\nThe current scenario: {room_data['start_scenario']}")
    prompt_lines.append("\nHere is the recent chat history (max 10):")
    
    sorted_messages = sorted(room_data.get('messages', []), key=lambda m: m.get('timestamp'))
    for msg in sorted_messages[-10:]:
        sender = msg['user_id']
        if sender == room_data['bot_name']: sender_display = "You"
        elif sender == "System": sender_display = "System"
        else: sender_display = next((data['nickname'] for uid, data in room_data.get('users', {}).items() if uid == sender), sender)
        prompt_lines.append(f"{sender_display}: {msg['text']}")

    prompt_lines.append(f"\n--- NEW MESSAGE ---")
    prompt_lines.append(f"{user_nickname}: {message_text}")
    prompt_lines.append("\n--- YOUR TASK ---")
    prompt_lines.append(
        "1. Respond in character to the new message.\n"
        f"2. Evaluate affection change for {user_nickname} ONLY (-20 to 20).\n"
        f"Difficulty: {room_data['difficulty']}/10.\n"
        "IMPORTANT: Return ONLY valid JSON. Do not use Markdown formatting.\n"
        "Format: {\"response\": \"your text response\", \"affection_change\": integer}"
    )
    return "\n".join(prompt_lines)

def build_openrouter_prompt_messages(room_data, user_nickname, message_text):
    # ... (Keep existing logic) ...
    system_prompt = (
        f"You are {room_data['bot_name']}. "
        f"Personality: {room_data['bot_personality']}. "
        f"Chatting privately with the winner, {user_nickname}."
    )
    messages = [{"role": "system", "content": system_prompt}]
    
    sorted_messages = sorted(room_data.get('messages', []), key=lambda m: m.get('timestamp'))
    for msg in sorted_messages[-15:]:
        if msg['user_id'] == room_data['bot_name']:
            messages.append({"role": "assistant", "content": msg['text']})
        elif msg['user_id'] != 'System':
            messages.append({"role": "user", "content": msg['text']})
    
    messages.append({"role": "user", "content": message_text})
    return messages

def parse_gemini_response(text):
    # ... (Keep existing logic) ...
    try:
        text = text.replace("```json", "").replace("```", "").strip()
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match: 
            return {"response": text, "affection_change": 0}
        data = json.loads(match.group(0))
        aff_change = data.get("affection_change", 0)
        if isinstance(aff_change, (dict, list, str)):
            try: aff_change = int(aff_change)
            except: aff_change = 0
        return {"response": data.get("response", "..."), "affection_change": int(aff_change)}
    except Exception as e:
        clean_text = text.replace('{"response":', '').replace('}', '').replace('"', '')
        return {"response": clean_text, "affection_change": 0}

# --- Routes ---

@app.route("/")
def index(): return redirect(url_for('public_bots'))

@app.route("/login")
def login(): return render_template("login.html")

@app.route("/register")
def register(): return render_template("register.html")

@app.route("/set-token", methods=["POST"])
def set_token():
    data = request.get_json()
    session['id_token'] = data['token']
    return jsonify({"success": True})

@app.route("/logout")
def logout():
    session.pop('id_token', None)
    flash("Logged out.")
    return redirect(url_for('login'))

# --- Routes: Public Bots, My Bots, Create, etc. (Keep exactly as they were) ---
@app.route("/public-bots")
def public_bots():
    bots = []
    active_sessions = {}
    try:
        bots_ref = db.collection('rooms').where('is_public', '==', True).limit(100)
        for doc in bots_ref.stream():
            bot_data = doc.to_dict()
            if bot_data.get('is_hosted') == True: continue
            bot_data['id'] = doc.id
            bots.append(bot_data)
    except Exception as e: print(f"Error: {e}")

    id_token = session.get('id_token')
    if id_token:
        try:
            decoded_token = auth.verify_id_token(id_token)
            user_uid = decoded_token['uid']
            my_instances = db.collection('rooms').where('owner_uid', '==', user_uid).where('is_hosted', '==', True).stream()
            for doc in my_instances:
                data = doc.to_dict()
                if data.get('parent_bot_id'): active_sessions[data['parent_bot_id']] = doc.id
        except: pass

    response = make_response(render_template("public_bots.html", bots=bots, active_sessions=active_sessions))
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response

@app.route("/my-bots")
@login_required 
def my_bots():
    bots = []
    user_uid = g.user_uid 
    try:
        bots_ref = db.collection('rooms').where('owner_uid', '==', user_uid).limit(50)
        for doc in bots_ref.stream():
            bot_data = doc.to_dict()
            if bot_data.get('is_hosted') == True: continue
            messages = bot_data.get('messages', [])
            if messages and len(messages) > 0 and "Instance created" in messages[0].get('text', ''): continue
            bot_data['id'] = doc.id
            bots.append(bot_data)
    except Exception as e: print(f"Error: {e}")
    return render_template("my_bots.html", bots=bots)

@app.route("/create", methods=["GET", "POST"])
@login_required 
def create_room():
    if request.method == "POST":
        room_code = generate_room_code()
        user_uid = g.user_uid 
        user_name = g.user.get('name', g.user.get('email', 'Anonymous'))
        py_time = datetime.utcnow().replace(tzinfo=timezone.utc) 
        
        new_room = {
            'bot_name': request.form.get('bot_name', 'Bot'),
            'bot_personality': request.form.get('bot_personality', 'friendly'),
            'start_scenario': request.form.get('start_scenario', 'Hello'),
            'difficulty': int(request.form.get('difficulty', 5)),
            'game_over': False,
            'users': {}, 
            'messages': [
                {'user_id': 'System', 'text': f"Bot created by {user_name}.", 'timestamp': py_time},
                {'user_id': request.form.get('bot_name'), 'text': request.form.get('start_scenario'), 'timestamp': py_time}
            ],
            'bot_image_url': request.form.get('bot_image_url'),
            'bot_appearance': request.form.get('appearance'),
            'owner_uid': user_uid,
            'owner_display_name': user_name,
            'is_public': request.form.get('is_public') == 'on',
            'model_version': 'flash',
            'is_hosted': False 
        }
        db.collection('rooms').document(room_code).set(new_room)
        return redirect(url_for('my_bots'))
    return render_template("create.html")

@app.route("/join", methods=["GET", "POST"])
@login_required 
def join_room():
    room_code = request.args.get('code') or request.form.get('room_code')
    mode = request.args.get('mode') or request.form.get('mode')
    privacy = request.form.get('privacy', 'public')
    initial_model = request.form.get('initial_model', 'flash')

    if not room_code: return redirect(url_for('index'))
    room_ref = db.collection('rooms').document(room_code)
    room_doc = room_ref.get()
    if not room_doc.exists:
        flash("Room not found.")
        return redirect(url_for('index'))
    
    room_data = room_doc.to_dict()
    user_uid = g.user_uid
    user_name = g.user.get('name', g.user.get('email', 'Player'))
    
    if mode == 'new':
        new_code = generate_room_code()
        py_time = datetime.utcnow().replace(tzinfo=timezone.utc)
        is_public_game = (privacy == 'public')
        chosen_model = 'flash' if is_public_game else initial_model
        fixed_model = not is_public_game

        cloned_data = {
            'bot_name': room_data.get('bot_name'),
            'bot_personality': room_data.get('bot_personality'),
            'start_scenario': room_data.get('start_scenario'),
            'difficulty': room_data.get('difficulty', 5),
            'game_over': False,
            'winner_uid': None,
            'users': { user_uid: {'nickname': user_name, 'score': 0} }, 
            'messages': [
                {'user_id': 'System', 'text': f"Instance created. Mode: {privacy.capitalize()}.", 'timestamp': py_time},
                {'user_id': room_data.get('bot_name'), 'text': room_data.get('start_scenario'), 'timestamp': py_time}
            ],
            'bot_image_url': room_data.get('bot_image_url'),
            'bot_appearance': room_data.get('bot_appearance'),
            'owner_uid': user_uid,
            'owner_display_name': user_name,
            'is_public': is_public_game,
            'model_version': chosen_model,
            'fixed_model': fixed_model, 
            'allowed_viewers': [],
            'parent_bot_id': room_code,
            'is_hosted': True 
        }
        db.collection('rooms').document(new_code).set(cloned_data)
        return redirect(url_for('chat_room', room_code=new_code))

    if user_uid not in room_data.get('users', {}):
        py_time = datetime.utcnow().replace(tzinfo=timezone.utc)
        room_ref.update({
            f'users.{user_uid}': {'nickname': user_name, 'score': 0}, 
            'messages': firestore.ArrayUnion([{
                'user_id': 'System', 'text': f"{user_name} joined.", 'timestamp': py_time
            }])
        })
    return redirect(url_for('chat_room', room_code=room_code))

# ==========================================
# === MODIFIED: Chat Room (Coin Logic) ===
# ==========================================
@app.route("/room/<room_code>", methods=["GET", "POST"])
@login_required
def chat_room(room_code):
    room_ref = db.collection('rooms').document(room_code)
    room_doc = room_ref.get()

    if not room_doc.exists: return redirect(url_for('index'))
    
    room_data = room_doc.to_dict()
    user_uid = g.user_uid
    
    if user_uid not in room_data.get('users', {}):
        return redirect(url_for('join_room', code=room_code))
        
    user_data = room_data['users'][user_uid]
    nickname = user_data.get('nickname', 'Player')
    is_owner = (room_data.get('owner_uid') == user_uid)
    is_winner = (room_data.get('winner_uid') == user_uid)

    has_access = True
    winner_uid = room_data.get('winner_uid')
    if winner_uid and user_uid != winner_uid and user_uid not in room_data.get('allowed_viewers', []):
        has_access = False

    # --- COIN CHECK 1: Get Current Wallet ---
    wallet = get_or_create_wallet(user_uid)
    current_coins = wallet.get('coins', 0)

    if request.method == "POST":
        if not has_access: return jsonify({'success': False, 'error': 'Locked'}), 403
        
        # --- COIN CHECK 2: Prevent Chat if Empty ---
        if current_coins < 1:
             return jsonify({'success': False, 'error': 'Out of coins! Buy more to continue.'}), 402
        
        msg = request.form.get('message')
        if not msg: return jsonify({'success': False}), 400
            
        try:
            py_time = datetime.utcnow().replace(tzinfo=timezone.utc)
            msgs_to_add = [{'user_id': user_uid, 'text': msg, 'timestamp': py_time}]
            
            if room_data.get('model_version') == 'pro':
                if not openrouter_api_key: raise ValueError("No OpenRouter Key")
                hist = build_openrouter_prompt_messages(room_data, nickname, msg)
                r = requests.post(
                    OPENROUTER_API_URL, 
                    headers={"Authorization": f"Bearer {openrouter_api_key}"},
                    json={"model": PRO_MODEL_NAME, "messages": hist}
                )
                bot_resp = r.json()['choices'][0]['message']['content']
                msgs_to_add.append({'user_id': room_data['bot_name'], 'text': bot_resp, 'timestamp': py_time})
                room_ref.update({'messages': firestore.ArrayUnion(msgs_to_add)})
            else:
                prompt = build_gemini_prompt(room_data, nickname, msg)
                res = model_flash.generate_content(prompt)
                parsed = parse_gemini_response(res.text)
                
                new_score = max(0, min(100, user_data['score'] + parsed['affection_change']))
                update_ops = { f'users.{user_uid}.score': new_score }
                
                msgs_to_add.append({'user_id': room_data['bot_name'], 'text': parsed['response'], 'timestamp': py_time})
                
                if new_score >= 100 and not winner_uid:
                    update_ops['winner_uid'] = user_uid
                    update_ops['winner_name'] = nickname
                    update_ops['allowed_viewers'] = [user_uid]
                    if not room_data.get('fixed_model', False):
                        update_ops['model_version'] = 'pro'
                        msgs_to_add.append({'user_id': 'System', 'text': f"🏆 {nickname} won! Unlocked Pro Model.", 'timestamp': py_time})
                    else:
                        msgs_to_add.append({'user_id': 'System', 'text': f"🏆 {nickname} won! You have conquered this timeline.", 'timestamp': py_time})

                update_ops['messages'] = firestore.ArrayUnion(msgs_to_add)
                room_ref.update(update_ops)

            # --- COIN DEDUCTION ---
            # Deduct 1 coin after successful generation
            db.collection('users').document(user_uid).update({
                'coins': firestore.Increment(-1)
            })
            current_coins -= 1
            
            return jsonify({'success': True, 'new_coin_count': current_coins})

        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    return render_template("room.html", 
                           room=room_data, 
                           room_code=room_code, 
                           user_id=user_uid, 
                           is_owner=is_owner,
                           is_winner=is_winner,
                           has_access=has_access)

# ==========================================
# === NEW: PAYHERE PAYMENT ROUTES ===
# ==========================================
@app.route('/initiate-payment', methods=['POST'])
@login_required
def initiate_payment():
    try:
        # Generate Unique Order ID
        order_id = f"OID_{generate_room_code(6)}" 
        amount = 1.00  # 1 USD
        currency = "USD" 
        
        # Generate PayHere Signature
        hash_val = generate_payhere_hash(
            PAYHERE_MERCHANT_ID, 
            order_id, 
            amount, 
            currency, 
            PAYHERE_MERCHANT_SECRET
        )

        # Return data for Frontend Form
        payment_data = {
            "action_url": PAYHERE_URL,
            "merchant_id": PAYHERE_MERCHANT_ID,
            "return_url": f"{DOMAIN_URL}/payment/return",
            "cancel_url": f"{DOMAIN_URL}/payment/cancel",
            "notify_url": f"{DOMAIN_URL}/payment/notify", 
            "order_id": order_id,
            "items": "20 Chat Coins",
            "currency": currency,
            "amount": amount,
            "first_name": g.user.get('name', 'User'),
            "last_name": "",
            "email": g.user.get('email', 'user@example.com'),
            "address": "Colombo",
            "city": "Colombo",
            "country": "Sri Lanka",
            "hash": hash_val,
            "custom_1": g.user_uid 
        }
        return jsonify(payment_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/payment/notify', methods=['POST'])
def payhere_notify():
    # PayHere sends Form Data
    merchant_id = request.form.get('merchant_id')
    order_id = request.form.get('order_id')
    payhere_amount = request.form.get('payhere_amount')
    payhere_currency = request.form.get('payhere_currency')
    status_code = request.form.get('status_code')
    md5sig = request.form.get('md5sig')
    custom_1 = request.form.get('custom_1') # user_uid

    # 1. Validate Signature
    local_md5sig = generate_payhere_hash(
        merchant_id, order_id, payhere_amount, payhere_currency, PAYHERE_MERCHANT_SECRET
    )

    if local_md5sig != md5sig:
        return "Signature Mismatch", 400

    # 2. Credit Coins (Status 2 = Success)
    if status_code == '2':
        user_uid = custom_1
        if user_uid:
            db.collection('users').document(user_uid).update({
                'coins': firestore.Increment(20)
            })
            print(f"PAYMENT: Credited 20 coins to {user_uid}")
            
    return "Accepted", 200

@app.route('/payment/return')
def payment_return():
    flash("Payment successful! 20 coins added.")
    return redirect(url_for('my_bots'))

@app.route('/payment/cancel')
def payment_cancel():
    flash("Payment cancelled.")
    return redirect(url_for('my_bots'))

# --- Image Generation Routes (Unchanged) ---
@app.route("/generate-bot-image", methods=["POST"])
@login_required
def generate_bot_image():
    if not stability_api_key: return jsonify({'success': False, 'error': 'No API Key'}), 500
    try:
        data = request.get_json()
        prompt = f"A beautiful portrait of a {data.get('age')} year old {data.get('gender')}, {data.get('appearance')}. digital art, anime style"
        answers = stability_api.generate(
            prompt=[generation.Prompt(text=prompt, parameters=generation.PromptParameters(weight=1.0))],
            style_preset="anime", steps=30, cfg_scale=7.0, width=512, height=512, samples=1
        )
        for resp in answers:
            for artifact in resp.artifacts:
                if artifact.type == generation.ARTIFACT_IMAGE:
                    upload_result = cloudinary.uploader.upload(artifact.binary, folder="bot_avatars", public_id=f"bot_{generate_room_code(10)}")
                    return jsonify({'success': True, 'image_url': upload_result.get('secure_url')})
        return jsonify({'success': False, 'error': 'No image generated'}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route("/upload-bot-image", methods=["POST"])
@login_required
def upload_bot_image():
    if 'file' not in request.files: return jsonify({'success': False, 'error': 'No file'}), 400
    file = request.files['file']
    if file:
        try:
            res = cloudinary.uploader.upload(file, folder="bot_avatars", public_id=f"bot_{generate_room_code(10)}")
            return jsonify({'success': True, 'image_url': res.get('secure_url')})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    return jsonify({'success': False}), 400

# --- Host Actions (Reset, Request, Delete) (Unchanged) ---
@app.route("/reset-chat", methods=["POST"])
@login_required
def reset_chat():
    code = request.form.get('room_code')
    room_ref = db.collection('rooms').document(code)
    doc = room_ref.get()
    if not doc.exists: return jsonify({'success': False}), 404
    room_data = doc.to_dict()
    if room_data.get('owner_uid') != g.user_uid: return jsonify({'success': False, 'error': 'No permission'}), 403
    
    if room_data.get('is_hosted', False):
        room_ref.delete()
        flash("Session ended.")
        return redirect(url_for('public_bots'))
    else:
        py_time = datetime.utcnow().replace(tzinfo=timezone.utc)
        updates = {
            'messages': [
                {'user_id': 'System', 'text': "Chat reset by Owner.", 'timestamp': py_time},
                {'user_id': room_data.get('bot_name'), 'text': room_data.get('start_scenario'), 'timestamp': py_time}
            ],
            'winner_uid': None, 'winner_name': None, 'model_version': 'flash', 'allowed_viewers': []
        }
        for uid in room_data.get('users', {}): updates[f'users.{uid}.score'] = 0
        room_ref.update(updates)
        return redirect(url_for('chat_room', room_code=code))

@app.route("/request-access", methods=["POST"])
@login_required
def request_access():
    code = request.form.get('room_code')
    name = g.user.get('name', 'User')
    db.collection('rooms').document(code).update({
        'access_requests': firestore.ArrayUnion([{'uid': g.user_uid, 'name': name}])
    })
    return jsonify({'success': True})

@app.route("/grant-access", methods=["POST"])
@login_required
def grant_access():
    code = request.form.get('room_code')
    target = request.form.get('target_uid')
    room_ref = db.collection('rooms').document(code)
    if room_ref.get().to_dict().get('winner_uid') != g.user_uid: return jsonify({'success': False}), 403
    room_ref.update({'allowed_viewers': firestore.ArrayUnion([target])})
    return jsonify({'success': True})

@app.route("/delete-bot", methods=["POST"])
@login_required
def delete_bot():
    room_code = request.form.get('room_code')
    room_ref = db.collection('rooms').document(room_code)
    doc = room_ref.get()
    if doc.exists and doc.to_dict().get('owner_uid') == g.user_uid:
        room_ref.delete()
        flash("Bot deleted.")
    return redirect(url_for('my_bots'))

if __name__ == "__main__":
    app.run(debug=True, port=5000)