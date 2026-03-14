"""
Trade AI Mentor - Complete Web Application with Subscriptions and Repair Guidance
Supports: Construction, Plumbing (Sewer), Electrical
Run with: python unified_trade_app.py
"""

from flask import Flask, request, render_template_string, jsonify, session, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import base64
import os
import json
from datetime import datetime, timedelta
import hashlib
import sqlite3
import stripe
from dotenv import load_dotenv
from functools import wraps
import secrets
from flask_mail import Mail, Message
import secrets
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', secrets.token_hex(32))

# ============================================
# STRIPE CONFIGURATION
# ============================================
stripe.api_key = os.getenv('STRIPE_SECRET_KEY')
STRIPE_PUBLISHABLE_KEY = os.getenv('STRIPE_PUBLISHABLE_KEY')
STRIPE_WEBHOOK_SECRET = os.getenv('STRIPE_WEBHOOK_SECRET')

# Subscription prices (create these in Stripe dashboard)
PRICE_IDS = {
    'monthly': os.getenv('STRIPE_MONTHLY_PRICE_ID'),
    'yearly': os.getenv('STRIPE_YEARLY_PRICE_ID'),
    'lifetime': os.getenv('STRIPE_LIFETIME_PRICE_ID')
}

# ============================================
# MAIL CONFIGURATION (for password reset)
# ============================================
app.config['MAIL_SERVER'] = 'smtp.gmail.com'  # or your email provider
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')  # your email
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')  # your app password
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_USERNAME')

mail = Mail(app)

# For generating secure tokens
serializer = URLSafeTimedSerializer(app.secret_key)

# ============================================
# DATABASE SETUP
# ============================================
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    # Create users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  email TEXT UNIQUE NOT NULL,
                  password_hash TEXT NOT NULL,
                  name TEXT,
                  subscription_status TEXT DEFAULT 'free',
                  subscription_end DATE,
                  free_trials_used INTEGER DEFAULT 0,
                  max_free_trials INTEGER DEFAULT 5,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  stripe_customer_id TEXT)''')
    
    # Create analyses table
    c.execute('''CREATE TABLE IF NOT EXISTS analyses
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  trade TEXT,
                  image_name TEXT,
                  prediction TEXT,
                  confidence REAL,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    # Create payment history table
    c.execute('''CREATE TABLE IF NOT EXISTS payments
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  stripe_session_id TEXT UNIQUE,
                  stripe_customer_id TEXT,
                  amount REAL,
                  currency TEXT,
                  status TEXT,
                  subscription_type TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    conn.commit()
    conn.close()

init_db()

# ============================================
# FLASK-LOGIN SETUP
# ============================================
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id, email, name, subscription_status, subscription_end, free_trials_used, max_free_trials):
        self.id = id
        self.email = email
        self.name = name
        self.subscription_status = subscription_status
        self.subscription_end = subscription_end
        self.free_trials_used = free_trials_used
        self.max_free_trials = max_free_trials
    
    def is_subscribed(self):
        if self.subscription_status == 'lifetime':
            return True
        if self.subscription_status == 'active' and self.subscription_end:
            if datetime.strptime(self.subscription_end, '%Y-%m-%d').date() >= datetime.now().date():
                return True
        return False
    
    def can_use_free_trial(self):
        return self.free_trials_used < self.max_free_trials
    
    def increment_free_trials(self):
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('UPDATE users SET free_trials_used = ? WHERE id = ?', 
                 (self.free_trials_used + 1, self.id))
        conn.commit()
        conn.close()
        self.free_trials_used += 1

@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    user = c.execute('SELECT * FROM users WHERE id = ?', (user_id,)).fetchone()
    conn.close()
    if user:
        return User(user[0], user[1], user[3], user[4], user[5], user[6], user[7])
    return None

# ============================================
# SUBSCRIPTION DECORATOR
# ============================================
def subscription_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            flash('Please login to continue', 'warning')
            return redirect(url_for('login', next=request.url))
        
        if not current_user.is_subscribed() and not current_user.can_use_free_trial():
            flash('Free trials exhausted. Please subscribe to continue.', 'warning')
            return redirect(url_for('pricing'))
        
        return f(*args, **kwargs)
    return decorated_function

# ============================================
# LOAD ALL THREE MODELS
# ============================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Loading models on {DEVICE}...")

# Model paths
MODEL_PATHS = {
    'construction': 'final_models/construction_defect_detector_best.pth',
    'plumbing': 'final_models/pro_physical_work_ai_mentor_best.pth',
    'electrical': 'final_models/electrical_defect_detector_best.pth'
}

# Class names for each trade
CLASS_NAMES = {
    'construction': ['normal', 'crack'],
    'plumbing': ['normal', 'crack', 'root', 'deposit', 'obstacle', 
                 'deformation', 'joint_displacement', 'hole', 'corrosion', 'infiltration'],
    'electrical': ['normal', 'burnt_wiring', 'corrosion', 'loose_connection',
                   'overheating', 'water_damage', 'cracked_insulation',
                   'exposed_wire', 'improper_wiring']
}

# Model accuracies
MODEL_ACCURACIES = {
    'construction': 99.47,
    'plumbing': 91.53,
    'electrical': 91.73
}

# Repair guidance for each defect type
REPAIR_GUIDANCE = {
    'construction': {
        'crack': {
            'steps': [
                'Clean the crack thoroughly with a wire brush',
                'Apply concrete bonding adhesive',
                'Fill with epoxy injection or polyurethane foam',
                'Smooth surface with putty knife',
                'Allow to cure for 24-48 hours'
            ],
            'tools': ['Wire brush', 'Epoxy kit', 'Putty knife', 'Safety gloves'],
            'safety': 'Wear eye protection and gloves. Ensure proper ventilation.'
        },
        'normal': {
            'steps': ['Surface appears sound', 'Monitor for any changes over time'],
            'tools': ['Camera', 'Notebook'],
            'safety': 'Normal construction site safety applies.'
        }
    },
    'plumbing': {
        'crack': {
            'steps': [
                'Turn off water supply immediately',
                'Clean and dry the area',
                'Apply pipe repair epoxy or clamp',
                'Replace damaged section if severe'
            ],
            'tools': ['Pipe repair clamp', 'Epoxy', 'Pipe wrench', 'Teflon tape'],
            'safety': 'Water damage can cause electrical hazards. Ensure area is dry.'
        },
        'root': {
            'steps': [
                'Use mechanical auger to cut roots',
                'Apply root-killing chemicals',
                'Consider professional hydro-jetting',
                'Schedule regular maintenance'
            ],
            'tools': ['Plumbing auger', 'Root killer', 'Safety goggles'],
            'safety': 'Chemicals can be harmful. Wear protective gear.'
        },
        'deposit': {
            'steps': [
                'Test water hardness',
                'Use descaling solution',
                'Install water softener if recurring',
                'Flush system thoroughly'
            ],
            'tools': ['Descaling kit', 'Pipe brush', 'Water test kit'],
            'safety': 'Descaling chemicals are corrosive. Use gloves.'
        },
        'obstacle': {
            'steps': [
                'Use plumber\'s snake to clear',
                'Check for foreign objects',
                'Consider camera inspection',
                'Install drain strainer'
            ],
            'tools': ['Plumber\'s snake', 'Drain camera', 'Shop vac'],
            'safety': 'Never use chemical drain cleaners with standing water.'
        },
        'deformation': {
            'steps': [
                'Assess extent of damage',
                'Cut out deformed section',
                'Replace with new pipe',
                'Check for underlying cause'
            ],
            'tools': ['Pipe cutter', 'Replacement pipe', 'Torch (for copper)', 'Fittings'],
            'safety': 'Gas torch requires fire safety precautions.'
        },
        'joint_displacement': {
            'steps': [
                'Loosen adjacent fittings',
                'Realign pipe sections',
                'Tighten and seal joints',
                'Test for leaks'
            ],
            'tools': ['Pipe wrench', 'Channel locks', 'Pipe dope', 'Teflon tape'],
            'safety': 'Ensure proper support before loosening.'
        },
        'hole': {
            'steps': [
                'Isolate the section',
                'Cut out damaged area',
                'Install patch or replace section',
                'Test with water pressure'
            ],
            'tools': ['Pipe cutter', 'Patch kit', 'Replacement pipe', 'Primer/cement'],
            'safety': 'Wear gloves when handling sharp edges.'
        },
        'corrosion': {
            'steps': [
                'Clean with wire brush',
                'Apply rust converter',
                'Paint with anti-corrosion coating',
                'Replace if structural integrity compromised'
            ],
            'tools': ['Wire brush', 'Rust converter', 'Safety mask', 'Paint brush'],
            'safety': 'Rust dust can be harmful. Wear mask.'
        },
        'infiltration': {
            'steps': [
                'Identify water source',
                'Seal entry points with hydraulic cement',
                'Apply waterproof membrane',
                'Improve drainage'
            ],
            'tools': ['Hydraulic cement', 'Trowel', 'Waterproofing membrane', 'Flashlight'],
            'safety': 'Wet areas can be slippery. Wear non-slip boots.'
        },
        'normal': {
            'steps': ['Pipe appears normal', 'Monitor for any changes'],
            'tools': ['Camera', 'Flashlight'],
            'safety': 'Standard plumbing safety applies.'
        }
    },
    'electrical': {
        'burnt_wiring': {
            'steps': [
                '⚠️ TURN OFF POWER AT BREAKER ⚠️',
                'Cut out damaged wire section',
                'Strip wire ends properly',
                'Connect with wire nuts',
                'Replace any melted outlets/switches'
            ],
            'tools': ['Wire cutters', 'Wire strippers', 'Wire nuts', 'Voltage tester'],
            'safety': '⚠️ ELECTRICAL SHOCK HAZARD - Ensure power is OFF and verify with tester!'
        },
        'corrosion': {
            'steps': [
                'Turn off power',
                'Clean contacts with wire brush',
                'Apply dielectric grease',
                'Replace severely corroded components'
            ],
            'tools': ['Wire brush', 'Dielectric grease', 'Contact cleaner', 'Safety glasses'],
            'safety': 'Corrosion dust can be toxic. Wear mask and gloves.'
        },
        'loose_connection': {
            'steps': [
                'Turn off power',
                'Tighten terminal screws',
                'Check for damaged wires',
                'Replace if wire is nicked'
            ],
            'tools': ['Screwdriver', 'Wire strippers', 'Voltage tester'],
            'safety': 'Loose connections cause fires. Ensure tightness.'
        },
        'overheating': {
            'steps': [
                '⚠️ TURN OFF POWER IMMEDIATELY ⚠️',
                'Check for overloaded circuit',
                'Replace damaged breaker',
                'Upgrade wire gauge if needed'
            ],
            'tools': ['Infrared thermometer', 'Screwdriver', 'Replacement breaker'],
            'safety': '🔥 FIRE HAZARD - Do not use until repaired!'
        },
        'water_damage': {
            'steps': [
                '⚠️ TURN OFF POWER AT MAIN ⚠️',
                'Dry completely with fan',
                'Replace affected outlets',
                'Check for rust damage'
            ],
            'tools': ['Fan/dehumidifier', 'Outlet tester', 'Screwdriver'],
            'safety': '💧 WATER + ELECTRICITY = DEATH - Ensure completely dry!'
        },
        'cracked_insulation': {
            'steps': [
                'Turn off power',
                'Wrap with electrical tape temporarily',
                'Replace damaged wire',
                'Use proper wire protection'
            ],
            'tools': ['Electrical tape', 'Wire cutters', 'Heat shrink tubing', 'Heat gun'],
            'safety': 'Exposed wires can shock. Treat as live until verified.'
        },
        'exposed_wire': {
            'steps': [
                '⚠️ TURN OFF POWER IMMEDIATELY ⚠️',
                'Cut and strip properly',
                'Reconnect with wire nut',
                'Ensure no copper shows'
            ],
            'tools': ['Wire cutters', 'Wire strippers', 'Wire nuts', 'Electrical tape'],
            'safety': '⚠️ IMMEDIATE SHOCK HAZARD - Turn off power NOW!'
        },
        'improper_wiring': {
            'steps': [
                'Document current wiring',
                'Consult wiring diagram',
                'Correct color coding',
                'Verify with multimeter'
            ],
            'tools': ['Multimeter', 'Wire markers', 'Screwdriver', 'Wiring diagram'],
            'safety': 'Incorrect wiring causes fires. Verify all connections.'
        },
        'normal': {
            'steps': ['Electrical system appears normal', 'Continue regular maintenance'],
            'tools': ['Voltage tester', 'Flashlight'],
            'safety': 'Always treat wires as live until verified.'
        }
    }
}

# Load all models
models_dict = {}

for trade, path in MODEL_PATHS.items():
    try:
        print(f"📦 Loading {trade} model...")
        checkpoint = torch.load(path, map_location=DEVICE)
        
        num_classes = len(CLASS_NAMES[trade])
        model = models.resnet18(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(DEVICE)
        model.eval()
        
        models_dict[trade] = model
        print(f"   ✅ {trade} model loaded ({MODEL_ACCURACIES[trade]}% accuracy)")
    except Exception as e:
        print(f"   ❌ Error loading {trade} model: {e}")
        models_dict[trade] = None

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============================================
# AUTHENTICATION ROUTES
# ============================================
@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        user = c.execute('SELECT id FROM users WHERE email = ?', (email,)).fetchone()
        conn.close()
        
        if user:
            # Generate reset token
            token = serializer.dumps(email, salt='password-reset-salt')
            
            # Create reset link
            reset_url = url_for('reset_password', token=token, _external=True)
            
            # Send email
            msg = Message('Password Reset Request - Trade AI Mentor',
                         recipients=[email])
            msg.body = f'''To reset your password, visit the following link:
{reset_url}

If you did not make this request, simply ignore this email and no changes will be made.
'''
            try:
                mail.send(msg)
                flash('Check your email for password reset instructions', 'success')
            except Exception as e:
                flash('Error sending email. Please try again.', 'danger')
                print(f"Email error: {e}")
        else:
            # Don't reveal if email exists or not (security)
            flash('If that email is registered, you will receive reset instructions', 'info')
        
        return redirect(url_for('login'))
    
    return render_template_string(FORGOT_PASSWORD_TEMPLATE)

@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    try:
        email = serializer.loads(token, salt='password-reset-salt', max_age=3600)  # 1 hour expiry
    except (SignatureExpired, BadSignature):
        flash('The password reset link is invalid or has expired', 'danger')
        return redirect(url_for('forgot_password'))
    
    if request.method == 'POST':
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return redirect(url_for('reset_password', token=token))
        
        if len(password) < 6:
            flash('Password must be at least 6 characters', 'danger')
            return redirect(url_for('reset_password', token=token))
        
        # Update password
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        password_hash = generate_password_hash(password)
        c.execute('UPDATE users SET password_hash = ? WHERE email = ?', 
                 (password_hash, email))
        conn.commit()
        conn.close()
        
        flash('Your password has been updated! You can now login.', 'success')
        return redirect(url_for('login'))
    
    return render_template_string(RESET_PASSWORD_TEMPLATE, token=token, email=email)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        name = request.form['name']
        
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        
        # Check if user exists
        existing = c.execute('SELECT id FROM users WHERE email = ?', (email,)).fetchone()
        if existing:
            conn.close()
            flash('Email already registered', 'danger')
            return redirect(url_for('register'))
        
        # Create user
        password_hash = generate_password_hash(password)
        c.execute('''INSERT INTO users (email, password_hash, name, subscription_status)
                     VALUES (?, ?, ?, ?)''', (email, password_hash, name, 'free'))
        user_id = c.lastrowid
        conn.commit()
        conn.close()
        
        # Log user in
        user = load_user(user_id)
        login_user(user)
        
        flash('Registration successful! You have 5 free trials.', 'success')
        return redirect(url_for('dashboard'))
    
    return render_template_string(REGISTER_TEMPLATE)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        user = c.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        conn.close()
        
        if user and check_password_hash(user[2], password):
            user_obj = User(user[0], user[1], user[3], user[4], user[5], user[6], user[7])
            login_user(user_obj)
            next_page = request.args.get('next')
            flash('Login successful!', 'success')
            return redirect(next_page or url_for('dashboard'))
        
        flash('Invalid email or password', 'danger')
    
    return render_template_string(LOGIN_TEMPLATE)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully', 'success')
    return redirect(url_for('home'))

@app.route('/dashboard')
@login_required
def dashboard():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # Get user's analysis history
    analyses = c.execute('''SELECT trade, image_name, prediction, confidence, created_at 
                           FROM analyses WHERE user_id = ? 
                           ORDER BY created_at DESC LIMIT 20''', (current_user.id,)).fetchall()
    conn.close()
    
    return render_template_string(DASHBOARD_TEMPLATE, 
                                 analyses=analyses,
                                 user=current_user)

# ============================================
# SUBSCRIPTION ROUTES
# ============================================
@app.route('/pricing')
def pricing():
    return render_template_string(PRICING_TEMPLATE, 
                                 stripe_key=STRIPE_PUBLISHABLE_KEY)

@app.route('/create-checkout-session', methods=['POST'])
@login_required
def create_checkout_session():
    try:
        plan = request.json.get('plan', 'monthly')
        
        # Create Stripe checkout session
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price': PRICE_IDS[plan],
                'quantity': 1,
            }],
            mode='subscription' if plan in ['monthly', 'yearly'] else 'payment',
            success_url=url_for('payment_success', _external=True) + '?session_id={CHECKOUT_SESSION_ID}',
            cancel_url=url_for('pricing', _external=True),
            client_reference_id=str(current_user.id),
            customer_email=current_user.email
        )
        
        return jsonify({'sessionId': checkout_session.id})
    except Exception as e:
        return jsonify({'error': str(e)}), 403

@app.route('/payment-success')
@login_required
def payment_success():
    session_id = request.args.get('session_id')
    
    if session_id:
        try:
            # Retrieve the session from Stripe
            checkout_session = stripe.checkout.Session.retrieve(session_id)
            
            # Update user's subscription
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            
            # Determine subscription type
            if checkout_session.mode == 'subscription':
                # Get subscription details
                subscription = stripe.Subscription.retrieve(checkout_session.subscription)
                plan_type = 'monthly' if 'month' in subscription.items.data[0].price.recurring.interval else 'yearly'
                end_date = datetime.now() + timedelta(days=30 if plan_type == 'monthly' else 365)
                
                c.execute('''UPDATE users 
                           SET subscription_status = ?, subscription_end = ?, stripe_customer_id = ?
                           WHERE id = ?''', 
                         ('active', end_date.strftime('%Y-%m-%d'), checkout_session.customer, current_user.id))
            else:
                # Lifetime purchase
                c.execute('''UPDATE users 
                           SET subscription_status = 'lifetime', stripe_customer_id = ?
                           WHERE id = ?''', 
                         (checkout_session.customer, current_user.id))
            
            # Record payment
            c.execute('''INSERT INTO payments (user_id, stripe_session_id, stripe_customer_id, 
                         amount, currency, status, subscription_type)
                         VALUES (?, ?, ?, ?, ?, ?, ?)''',
                     (current_user.id, session_id, checkout_session.customer,
                      checkout_session.amount_total / 100, checkout_session.currency,
                      'succeeded', 'subscription'))
            
            conn.commit()
            conn.close()
            
            flash('Payment successful! Your subscription is now active.', 'success')
        except Exception as e:
            flash(f'Error processing payment: {str(e)}', 'danger')
    
    return redirect(url_for('dashboard'))

@app.route('/webhook', methods=['POST'])
def webhook():
    payload = request.data
    sig_header = request.headers.get('Stripe-Signature')
    
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, STRIPE_WEBHOOK_SECRET
        )
    except ValueError:
        return 'Invalid payload', 400
    except stripe.error.SignatureVerificationError:
        return 'Invalid signature', 400
    
    # Handle the event
    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        user_id = session.get('client_reference_id')
        
        if user_id:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            
            # Update user subscription based on session
            if session.get('mode') == 'subscription':
                c.execute('''UPDATE users SET subscription_status = 'active' 
                           WHERE id = ?''', (user_id,))
            else:
                c.execute('''UPDATE users SET subscription_status = 'lifetime' 
                           WHERE id = ?''', (user_id,))
            
            conn.commit()
            conn.close()
    
    return '', 200

@app.route('/cancel-subscription', methods=['POST'])
@login_required
def cancel_subscription():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    user = c.execute('SELECT stripe_customer_id FROM users WHERE id = ?', 
                    (current_user.id,)).fetchone()
    conn.close()
    
    if user and user[0]:
        try:
            # Get active subscriptions
            subscriptions = stripe.Subscription.list(customer=user[0], status='active')
            for sub in subscriptions:
                stripe.Subscription.delete(sub.id)
            
            # Update database
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute('''UPDATE users SET subscription_status = 'cancelled' 
                       WHERE id = ?''', (current_user.id,))
            conn.commit()
            conn.close()
            
            flash('Subscription cancelled successfully', 'success')
        except Exception as e:
            flash(f'Error cancelling subscription: {str(e)}', 'danger')
    
    return redirect(url_for('dashboard'))

# ============================================
# AI PREDICTION ROUTES (Protected)
# ============================================
@app.route('/')
def home():
    return render_template_string(HOME_TEMPLATE)

@app.route('/analyze')
@login_required
@subscription_required
def analyze_page():
    return render_template_string(ANALYZE_TEMPLATE,
                                 models=models_dict.keys(),
                                 accuracies=MODEL_ACCURACIES,
                                 CLASS_NAMES=CLASS_NAMES)

@app.route('/trade_info/<trade>')
def trade_info(trade):
    if trade not in CLASS_NAMES:
        return jsonify({'error': 'Trade not found'}), 404
    
    return jsonify({
        'name': trade.capitalize(),
        'accuracy': MODEL_ACCURACIES[trade],
        'defects': CLASS_NAMES[trade]
    })

@app.route('/predict', methods=['POST'])
@login_required
@subscription_required
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    trade = request.form.get('trade', 'construction')
    if trade not in models_dict or models_dict[trade] is None:
        return jsonify({'error': f'Model for {trade} not available'}), 400
    
    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')
    
    # Track free trial usage
    if not current_user.is_subscribed():
        current_user.increment_free_trials()
    
    # Predict
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    model = models_dict[trade]
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
    
    prediction = CLASS_NAMES[trade][predicted_class]
    
    # Get repair guidance
    guidance = REPAIR_GUIDANCE[trade].get(prediction, REPAIR_GUIDANCE[trade].get('normal', {
        'steps': ['Consult a professional', 'Document the issue'],
        'tools': ['Camera', 'Notebook'],
        'safety': 'Always follow safety protocols.'
    }))
    
    # Save analysis to database
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''INSERT INTO analyses (user_id, trade, image_name, prediction, confidence)
                 VALUES (?, ?, ?, ?, ?)''',
              (current_user.id, trade, file.filename, prediction, 
               probabilities[predicted_class].item()))
    conn.commit()
    conn.close()
    
    # Prepare results with guidance
    result = {
        'prediction': prediction,
        'confidence': probabilities[predicted_class].item(),
        'free_trials_remaining': current_user.max_free_trials - current_user.free_trials_used if not current_user.is_subscribed() else None,
        'probabilities': {
            name: prob.item() 
            for name, prob in zip(CLASS_NAMES[trade], probabilities)
        },
        'guidance': guidance
    }
    
    return jsonify(result)

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'models': {
            trade: 'loaded' if model else 'error'
            for trade, model in models_dict.items()
        },
        'accuracies': MODEL_ACCURACIES
    })

# ============================================
# HTML TEMPLATES
# ============================================

HOME_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trade AI Mentor - Professional Defect Detection</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }
        .navbar {
            background: white;
            padding: 1rem 2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .logo {
            font-size: 1.5rem;
            font-weight: bold;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .nav-links a {
            margin-left: 2rem;
            text-decoration: none;
            color: #666;
            font-weight: 500;
            transition: color 0.3s;
        }
        .nav-links a:hover { color: #667eea; }
        .btn {
            padding: 0.5rem 1.5rem;
            border-radius: 25px;
            border: none;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.3s;
        }
        .btn-primary {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
        }
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        .container {
            max-width: 1200px;
            margin: 2rem auto;
            padding: 0 2rem;
        }
        .hero {
            text-align: center;
            color: white;
            padding: 4rem 0;
        }
        .hero h1 {
            font-size: 3.5rem;
            margin-bottom: 1rem;
        }
        .hero p {
            font-size: 1.2rem;
            opacity: 0.9;
            margin-bottom: 2rem;
        }
        .features {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 2rem;
            margin-top: 3rem;
        }
        .feature-card {
            background: white;
            border-radius: 15px;
            padding: 2rem;
            text-align: center;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }
        .feature-card:hover { transform: translateY(-5px); }
        .feature-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
        }
        .feature-card h3 {
            margin-bottom: 1rem;
            color: #333;
        }
        .feature-card p { color: #666; line-height: 1.6; }
        .accuracy-badge {
            display: inline-block;
            background: #48bb78;
            color: white;
            padding: 0.25rem 1rem;
            border-radius: 20px;
            font-size: 0.9rem;
            margin-top: 1rem;
        }
        .footer {
            text-align: center;
            color: rgba(255,255,255,0.8);
            margin-top: 4rem;
            padding: 2rem 0;
        }
        @media (max-width: 768px) {
            .hero h1 { font-size: 2.5rem; }
            .nav-links { display: none; }
        }
    </style>
</head>
<body>
    <nav class="navbar">
        <div class="logo">🔧 Trade AI Mentor</div>
        <div class="nav-links">
            {% if current_user.is_authenticated %}
                <a href="{{ url_for('dashboard') }}">Dashboard</a>
                <a href="{{ url_for('pricing') }}">Pricing</a>
                <a href="{{ url_for('logout') }}">Logout</a>
            {% else %}
                <a href="{{ url_for('login') }}">Login</a>
                <a href="{{ url_for('register') }}">Register</a>
            {% endif %}
        </div>
    </nav>
    
    <div class="container">
        <div class="hero">
            <h1>AI-Powered Defect Detection<br>for Tradespeople</h1>
            <p>Professional-grade AI that spots cracks, plumbing issues, and electrical hazards in seconds</p>
            {% if current_user.is_authenticated %}
                <a href="{{ url_for('analyze_page') }}">
                    <button class="btn btn-primary">Start Analyzing →</button>
                </a>
            {% else %}
                <a href="{{ url_for('register') }}">
                    <button class="btn btn-primary">Get Started Free →</button>
                </a>
            {% endif %}
        </div>
        
        <div class="features">
            <div class="feature-card">
                <div class="feature-icon">🏗️</div>
                <h3>Construction</h3>
                <p>Detect cracks in concrete, walls, and structures with 99.47% accuracy. Get instant repair guidance.</p>
                <span class="accuracy-badge">99.47% Accurate</span>
            </div>
            
            <div class="feature-card">
                <div class="feature-icon">🔧</div>
                <h3>Plumbing</h3>
                <p>Identify 10 different sewer defects including cracks, roots, deposits, and infiltration.</p>
                <span class="accuracy-badge">91.53% Accurate</span>
            </div>
            
            <div class="feature-card">
                <div class="feature-icon">⚡</div>
                <h3>Electrical</h3>
                <p>Spot burnt wiring, corrosion, loose connections, and 6 other electrical hazards.</p>
                <span class="accuracy-badge">91.73% Accurate</span>
            </div>
        </div>
        
        <div class="footer">
            <p>© 2026 Trade AI Mentor. All rights reserved.</p>
            <p style="margin-top: 0.5rem;">Trained on 150,000+ professional images</p>
        </div>
    </div>
</body>
</html>
'''

LOGIN_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login - Trade AI Mentor</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .auth-card {
            background: white;
            border-radius: 20px;
            padding: 3rem;
            width: 100%;
            max-width: 400px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        .auth-card h2 {
            margin-bottom: 2rem;
            color: #333;
            text-align: center;
        }
        .form-group {
            margin-bottom: 1.5rem;
        }
        .form-group label {
            display: block;
            margin-bottom: 0.5rem;
            color: #666;
            font-weight: 500;
        }
        .form-group input {
            width: 100%;
            padding: 0.75rem;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 1rem;
            transition: border-color 0.3s;
        }
        .form-group input:focus {
            outline: none;
            border-color: #667eea;
        }
        .btn {
            width: 100%;
            padding: 0.75rem;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        .footer-links {
            margin-top: 1.5rem;
            text-align: center;
        }
        .footer-links a {
            color: #667eea;
            text-decoration: none;
        }
        .footer-links a:hover {
            text-decoration: underline;
        }
        .flash-messages {
            margin-bottom: 1rem;
        }
        .flash-message {
            padding: 0.75rem;
            border-radius: 10px;
            margin-bottom: 0.5rem;
        }
        .flash-success { background: #c6f6d5; color: #22543d; }
        .flash-danger { background: #fed7d7; color: #742a2a; }
        .flash-warning { background: #feebc8; color: #744210; }
    </style>
</head>
<body>
    <div class="auth-card">
        <h2>🔧 Welcome Back</h2>
        
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                <div class="flash-messages">
                    {% for category, message in messages %}
                        <div class="flash-message flash-{{ category }}">{{ message }}</div>
                    {% endfor %}
                </div>
            {% endif %}
        {% endwith %}
        
        <form method="POST">
            <div class="form-group">
                <label>Email</label>
                <input type="email" name="email" required>
            </div>
            <div class="form-group">
                <label>Password</label>
                <input type="password" name="password" required>
            </div>
            <button type="submit" class="btn">Login</button>
        </form>
        
        <div class="footer-links">
            <a href="{{ url_for('register') }}">Don't have an account? Sign up</a>
            <br>
            <a href="{{ url_for('forgot_password') }}">Forgot Password?</a>
            <a href="{{ url_for('home') }}">← Back to Home</a>
        </div>
    </div>
</body>
</html>
'''

REGISTER_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sign Up - Trade AI Mentor</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .auth-card {
            background: white;
            border-radius: 20px;
            padding: 3rem;
            width: 100%;
            max-width: 400px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        .auth-card h2 {
            margin-bottom: 2rem;
            color: #333;
            text-align: center;
        }
        .form-group {
            margin-bottom: 1.5rem;
        }
        .form-group label {
            display: block;
            margin-bottom: 0.5rem;
            color: #666;
            font-weight: 500;
        }
        .form-group input {
            width: 100%;
            padding: 0.75rem;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 1rem;
            transition: border-color 0.3s;
        }
        .form-group input:focus {
            outline: none;
            border-color: #667eea;
        }
        .btn {
            width: 100%;
            padding: 0.75rem;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        .footer-links {
            margin-top: 1.5rem;
            text-align: center;
        }
        .footer-links a {
            color: #667eea;
            text-decoration: none;
        }
        .footer-links a:hover {
            text-decoration: underline;
        }
        .free-trial-badge {
            background: linear-gradient(135deg, #48bb78, #38a169);
            color: white;
            text-align: center;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1.5rem;
        }
    </style>
</head>
<body>
    <div class="auth-card">
        <h2>🚀 Start Your Free Trial</h2>
        
        <div class="free-trial-badge">
            <strong>5 Free Analyses</strong><br>
            No credit card required
        </div>
        
        <form method="POST">
            <div class="form-group">
                <label>Full Name</label>
                <input type="text" name="name" required>
            </div>
            <div class="form-group">
                <label>Email</label>
                <input type="email" name="email" required>
            </div>
            <div class="form-group">
                <label>Password</label>
                <input type="password" name="password" required minlength="6">
            </div>
            <button type="submit" class="btn">Create Account</button>
        </form>
        
        <div class="footer-links">
            <a href="{{ url_for('login') }}">Already have an account? Login</a>
            <br>
            <a href="{{ url_for('home') }}">← Back to Home</a>
        </div>
    </div>
</body>
</html>
'''

PRICING_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pricing - Trade AI Mentor</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
        }
        .navbar {
            background: white;
            padding: 1rem 2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .logo {
            font-size: 1.5rem;
            font-weight: bold;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .nav-links a {
            margin-left: 2rem;
            text-decoration: none;
            color: #666;
            font-weight: 500;
        }
        .nav-links a:hover { color: #667eea; }
        .container {
            max-width: 1200px;
            margin: 2rem auto;
            padding: 0 2rem;
        }
        .pricing-header {
            text-align: center;
            margin-bottom: 3rem;
        }
        .pricing-header h1 {
            font-size: 2.5rem;
            color: #333;
            margin-bottom: 1rem;
        }
        .pricing-header p {
            font-size: 1.2rem;
            color: #666;
        }
        .pricing-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 2rem;
            margin-top: 2rem;
        }
        .pricing-card {
            background: white;
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            transition: transform 0.3s;
            position: relative;
            overflow: hidden;
        }
        .pricing-card:hover {
            transform: translateY(-5px);
        }
        .popular-badge {
            position: absolute;
            top: 1rem;
            right: -2rem;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 0.5rem 3rem;
            transform: rotate(45deg);
            font-size: 0.9rem;
            font-weight: 600;
        }
        .pricing-card h3 {
            font-size: 1.8rem;
            color: #333;
            margin-bottom: 1rem;
        }
        .price {
            font-size: 2.5rem;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 1rem;
        }
        .price span {
            font-size: 1rem;
            color: #666;
            font-weight: normal;
        }
        .features {
            list-style: none;
            margin: 2rem 0;
        }
        .features li {
            padding: 0.5rem 0;
            color: #666;
        }
        .features li:before {
            content: "✓";
            color: #48bb78;
            font-weight: bold;
            margin-right: 0.5rem;
        }
        .btn {
            width: 100%;
            padding: 1rem;
            border: none;
            border-radius: 10px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }
        .btn-primary {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
        }
        .btn-outline {
            background: transparent;
            border: 2px solid #667eea;
            color: #667eea;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        .free-trial-note {
            text-align: center;
            margin-top: 2rem;
            color: #666;
        }
        .footer {
            text-align: center;
            margin-top: 4rem;
            padding: 2rem;
            color: #666;
        }
        #payment-message {
            color: rgb(105, 115, 134);
            font-size: 16px;
            line-height: 20px;
            padding-top: 12px;
            text-align: center;
        }
    </style>
    <script src="https://js.stripe.com/v3/"></script>
</head>
<body>
    <nav class="navbar">
        <div class="logo">🔧 Trade AI Mentor</div>
        <div class="nav-links">
            {% if current_user.is_authenticated %}
                <a href="{{ url_for('dashboard') }}">Dashboard</a>
                <a href="{{ url_for('analyze_page') }}">Analyze</a>
                <a href="{{ url_for('logout') }}">Logout</a>
            {% else %}
                <a href="{{ url_for('login') }}">Login</a>
                <a href="{{ url_for('register') }}">Register</a>
            {% endif %}
        </div>
    </nav>
    
    <div class="container">
        <div class="pricing-header">
            <h1>Simple, Transparent Pricing</h1>
            <p>Choose the plan that's right for you</p>
        </div>
        
        <div class="pricing-grid">
            <!-- Free Trial Card -->
            <div class="pricing-card">
                <h3>Free Trial</h3>
                <div class="price">$0</div>
                <ul class="features">
                    <li>5 free analyses</li>
                    <li>Access to all 3 trades</li>
                    <li>Basic repair guidance</li>
                    <li>No credit card required</li>
                </ul>
                {% if current_user.is_authenticated %}
                    <a href="{{ url_for('analyze_page') }}">
                        <button class="btn btn-outline">Start Free</button>
                    </a>
                {% else %}
                    <a href="{{ url_for('register') }}">
                        <button class="btn btn-outline">Sign Up Free</button>
                    </a>
                {% endif %}
            </div>
            
            <!-- Monthly Pro Card -->
            <div class="pricing-card">
                <div class="popular-badge">POPULAR</div>
                <h3>Monthly Pro</h3>
                <div class="price">$4.99<span>/month</span></div>
                <ul class="features">
                    <li>Unlimited analyses</li>
                    <li>All 3 trades</li>
                    <li>Full repair guidance</li>
                    <li>Analysis history</li>
                    <li>Priority support</li>
                </ul>
                <button class="btn btn-primary" onclick="handleCheckout('monthly')">Subscribe Now</button>
            </div>
            
            <!-- Yearly Pro Card -->
            <div class="pricing-card">
                <h3>Yearly Pro</h3>
                <div class="price">$39.99<span>/year</span></div>
                <p style="color: #48bb78; margin-top: -0.5rem;">Save 33%</p>
                <ul class="features">
                    <li>Everything in Monthly Pro</li>
                    <li>2 months free</li>
                    <li>Priority feature requests</li>
                </ul>
                <button class="btn btn-primary" onclick="handleCheckout('yearly')">Subscribe Now</button>
            </div>
            
            <!-- Lifetime Card -->
            <div class="pricing-card">
                <h3>Lifetime</h3>
                <div class="price">$79.99</div>
                <ul class="features">
                    <li>Never pay again</li>
                    <li>All future features</li>
                    <li>Lifetime updates</li>
                    <li>VIP support</li>
                </ul>
                <button class="btn btn-primary" onclick="handleCheckout('lifetime')">Buy Lifetime</button>
            </div>
        </div>
        
        <div class="free-trial-note">
            <p>All plans include access to Construction, Plumbing, and Electrical AI models.</p>
            <p style="margin-top: 0.5rem;">Questions? Contact us at support@tradeaimentor.com</p>
        </div>
        
        <div id="payment-message"></div>
    </div>
    
    <div class="footer">
        <p>© 2026 Trade AI Mentor. All rights reserved.</p>
    </div>
    
    <script>
        const stripe = Stripe('{{ stripe_key }}');
        
        async function handleCheckout(plan) {
            const response = await fetch('/create-checkout-session', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ plan: plan }),
            });
            
            const session = await response.json();
            
            if (session.error) {
                const message = document.getElementById('payment-message');
                message.textContent = session.error;
                message.style.color = '#e53e3e';
            } else {
                const result = await stripe.redirectToCheckout({
                    sessionId: session.sessionId,
                });
                
                if (result.error) {
                    const message = document.getElementById('payment-message');
                    message.textContent = result.error.message;
                    message.style.color = '#e53e3e';
                }
            }
        }
    </script>
</body>
</html>
'''

DASHBOARD_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard - Trade AI Mentor</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
        }
        .navbar {
            background: white;
            padding: 1rem 2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .logo {
            font-size: 1.5rem;
            font-weight: bold;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .nav-links a {
            margin-left: 2rem;
            text-decoration: none;
            color: #666;
            font-weight: 500;
        }
        .nav-links a:hover { color: #667eea; }
        .container {
            max-width: 1200px;
            margin: 2rem auto;
            padding: 0 2rem;
        }
        .welcome-card {
            background: white;
            border-radius: 20px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        .welcome-card h1 {
            color: #333;
            margin-bottom: 0.5rem;
        }
        .subscription-badge {
            display: inline-block;
            padding: 0.5rem 1.5rem;
            border-radius: 25px;
            font-weight: 600;
            margin-top: 1rem;
        }
        .badge-active { background: #c6f6d5; color: #22543d; }
        .badge-free { background: #fed7d7; color: #742a2a; }
        .badge-lifetime { background: #feebc8; color: #744210; }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 2rem 0;
        }
        .stat-card {
            background: white;
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .stat-number {
            font-size: 2rem;
            font-weight: bold;
            color: #667eea;
        }
        .stat-label {
            color: #666;
            margin-top: 0.5rem;
        }
        .analytics-table {
            background: white;
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        .analytics-table h2 {
            margin-bottom: 1rem;
            color: #333;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th {
            text-align: left;
            padding: 1rem 0.5rem;
            border-bottom: 2px solid #e0e0e0;
            color: #666;
        }
        td {
            padding: 1rem 0.5rem;
            border-bottom: 1px solid #e0e0e0;
        }
        .btn {
            display: inline-block;
            padding: 0.75rem 2rem;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 10px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            text-decoration: none;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        .btn-outline {
            background: transparent;
            border: 2px solid #667eea;
            color: #667eea;
        }
        .btn-outline:hover { background: #667eea; color: white; }
        .trial-progress {
            background: #e0e0e0;
            border-radius: 10px;
            height: 10px;
            margin: 1rem 0;
            overflow: hidden;
        }
        .trial-progress-fill {
            height: 100%;
            background: linear-gradient(135deg, #667eea, #764ba2);
            border-radius: 10px;
        }
    </style>
</head>
<body>
    <nav class="navbar">
        <div class="logo">🔧 Trade AI Mentor</div>
        <div class="nav-links">
            <a href="{{ url_for('dashboard') }}">Dashboard</a>
            <a href="{{ url_for('analyze_page') }}">New Analysis</a>
            <a href="{{ url_for('pricing') }}">Upgrade</a>
            <a href="{{ url_for('logout') }}">Logout</a>
        </div>
    </nav>
    
    <div class="container">
        <div class="welcome-card">
            <h1>Welcome back, {{ user.name }}! 👋</h1>
            
            {% if user.is_subscribed() %}
                <div class="subscription-badge badge-active">
                    ✨ Pro Subscriber
                    {% if user.subscription_status == 'lifetime' %}
                        (Lifetime)
                    {% elif user.subscription_end %}
                        until {{ user.subscription_end }}
                    {% endif %}
                </div>
            {% else %}
                <div class="subscription-badge badge-free">
                    🔍 Free Trial: {{ user.free_trials_used }}/{{ user.max_free_trials }} analyses used
                </div>
                <div class="trial-progress">
                    <div class="trial-progress-fill" style="width: {{ (user.free_trials_used / user.max_free_trials * 100) }}%"></div>
                </div>
                <a href="{{ url_for('pricing') }}" class="btn" style="margin-top: 1rem;">Upgrade to Pro →</a>
            {% endif %}
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">{{ analyses|length }}</div>
                <div class="stat-label">Total Analyses</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ analyses|selectattr('2', 'equalto', 'crack')|list|length }}</div>
                <div class="stat-label">Cracks Detected</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ analyses|selectattr('2', 'equalto', 'burnt_wiring')|list|length }}</div>
                <div class="stat-label">Electrical Issues</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ analyses|selectattr('2', 'equalto', 'root')|list|length }}</div>
                <div class="stat-label">Root Intrusions</div>
            </div>
        </div>
        
        <div class="analytics-table">
            <h2>📊 Recent Analysis History</h2>
            
            {% if analyses %}
                <table>
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Trade</th>
                            <th>Image</th>
                            <th>Prediction</th>
                            <th>Confidence</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for analysis in analyses %}
                            <tr>
                                <td>{{ analysis[4] }}</td>
                                <td>
                                    {% if analysis[0] == 'construction' %}🏗️ Construction
                                    {% elif analysis[0] == 'plumbing' %}🔧 Plumbing
                                    {% else %}⚡ Electrical
                                    {% endif %}
                                </td>
                                <td>{{ analysis[1]|truncate(20) }}</td>
                                <td>{{ analysis[2].replace('_', ' ').title() }}</td>
                                <td>{{ "%.1f"|format(analysis[3] * 100) }}%</td>
                            </tr>
                        {% endfor %}
                    </tbody>
                </table>
            {% else %}
                <p style="color: #666; text-align: center; padding: 2rem;">No analyses yet. Start your first analysis!</p>
            {% endif %}
            
            <div style="text-align: center; margin-top: 2rem;">
                <a href="{{ url_for('analyze_page') }}" class="btn btn-outline">New Analysis →</a>
            </div>
        </div>
    </div>
</body>
</html>
'''

FORGOT_PASSWORD_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Forgot Password - Trade AI Mentor</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .auth-card {
            background: white;
            border-radius: 20px;
            padding: 3rem;
            width: 100%;
            max-width: 400px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        .auth-card h2 {
            margin-bottom: 1rem;
            color: #333;
            text-align: center;
        }
        .auth-card p {
            color: #666;
            text-align: center;
            margin-bottom: 2rem;
        }
        .form-group {
            margin-bottom: 1.5rem;
        }
        .form-group label {
            display: block;
            margin-bottom: 0.5rem;
            color: #666;
            font-weight: 500;
        }
        .form-group input {
            width: 100%;
            padding: 0.75rem;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 1rem;
            transition: border-color 0.3s;
        }
        .form-group input:focus {
            outline: none;
            border-color: #667eea;
        }
        .btn {
            width: 100%;
            padding: 0.75rem;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        .footer-links {
            margin-top: 1.5rem;
            text-align: center;
        }
        .footer-links a {
            color: #667eea;
            text-decoration: none;
        }
        .footer-links a:hover {
            text-decoration: underline;
        }
        .flash-messages {
            margin-bottom: 1rem;
        }
        .flash-message {
            padding: 0.75rem;
            border-radius: 10px;
            margin-bottom: 0.5rem;
        }
        .flash-success { background: #c6f6d5; color: #22543d; }
        .flash-danger { background: #fed7d7; color: #742a2a; }
        .flash-info { background: #bee3f8; color: #2c5282; }
    </style>
</head>
<body>
    <div class="auth-card">
        <h2>🔐 Reset Password</h2>
        <p>Enter your email address and we'll send you instructions to reset your password.</p>
        
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                <div class="flash-messages">
                    {% for category, message in messages %}
                        <div class="flash-message flash-{{ category }}">{{ message }}</div>
                    {% endfor %}
                </div>
            {% endif %}
        {% endwith %}
        
        <form method="POST">
            <div class="form-group">
                <label>Email Address</label>
                <input type="email" name="email" required placeholder="your@email.com">
            </div>
            <button type="submit" class="btn">Send Reset Instructions</button>
        </form>
        
        <div class="footer-links">
            <a href="{{ url_for('login') }}">← Back to Login</a>
        </div>
    </div>
</body>
</html>
'''

RESET_PASSWORD_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reset Password - Trade AI Mentor</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .auth-card {
            background: white;
            border-radius: 20px;
            padding: 3rem;
            width: 100%;
            max-width: 400px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        .auth-card h2 {
            margin-bottom: 1rem;
            color: #333;
            text-align: center;
        }
        .auth-card p {
            color: #666;
            text-align: center;
            margin-bottom: 2rem;
        }
        .form-group {
            margin-bottom: 1.5rem;
        }
        .form-group label {
            display: block;
            margin-bottom: 0.5rem;
            color: #666;
            font-weight: 500;
        }
        .form-group input {
            width: 100%;
            padding: 0.75rem;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 1rem;
            transition: border-color 0.3s;
        }
        .form-group input:focus {
            outline: none;
            border-color: #667eea;
        }
        .btn {
            width: 100%;
            padding: 0.75rem;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        .footer-links {
            margin-top: 1.5rem;
            text-align: center;
        }
        .footer-links a {
            color: #667eea;
            text-decoration: none;
        }
        .footer-links a:hover {
            text-decoration: underline;
        }
        .flash-messages {
            margin-bottom: 1rem;
        }
        .flash-message {
            padding: 0.75rem;
            border-radius: 10px;
            margin-bottom: 0.5rem;
        }
        .flash-success { background: #c6f6d5; color: #22543d; }
        .flash-danger { background: #fed7d7; color: #742a2a; }
    </style>
</head>
<body>
    <div class="auth-card">
        <h2>🔐 Set New Password</h2>
        <p>For {{ email }}</p>
        
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                <div class="flash-messages">
                    {% for category, message in messages %}
                        <div class="flash-message flash-{{ category }}">{{ message }}</div>
                    {% endfor %}
                </div>
            {% endif %}
        {% endwith %}
        
        <form method="POST">
            <div class="form-group">
                <label>New Password</label>
                <input type="password" name="password" required minlength="6" placeholder="••••••">
            </div>
            <div class="form-group">
                <label>Confirm Password</label>
                <input type="password" name="confirm_password" required minlength="6" placeholder="••••••">
            </div>
            <button type="submit" class="btn">Update Password</button>
        </form>
        
        <div class="footer-links">
            <a href="{{ url_for('login') }}">← Back to Login</a>
        </div>
    </div>
</body>
</html>
'''

ANALYZE_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analyze - Trade AI Mentor</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
        }
        .navbar {
            background: white;
            padding: 1rem 2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .logo {
            font-size: 1.5rem;
            font-weight: bold;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .nav-links a {
            margin-left: 2rem;
            text-decoration: none;
            color: #666;
            font-weight: 500;
        }
        .nav-links a:hover { color: #667eea; }
        .container {
            max-width: 1200px;
            margin: 2rem auto;
            padding: 0 2rem;
        }
        .analyze-card {
            background: white;
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        .trade-selector {
            display: flex;
            gap: 1rem;
            margin-bottom: 2rem;
            flex-wrap: wrap;
        }
        .trade-btn {
            flex: 1;
            min-width: 150px;
            padding: 1rem;
            border: none;
            border-radius: 10px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            background: #f0f0f0;
            color: #333;
        }
        .trade-btn.active {
            color: white;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .trade-btn.construction.active { background: #f39c12; }
        .trade-btn.plumbing.active { background: #3498db; }
        .trade-btn.electrical.active { background: #e74c3c; }
        .model-info {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .accuracy-badge {
            background: #27ae60;
            color: white;
            padding: 0.25rem 1rem;
            border-radius: 20px;
        }
        .defect-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin: 1rem 0;
        }
        .defect-tag {
            background: #e0e0e0;
            padding: 0.25rem 0.75rem;
            border-radius: 15px;
            font-size: 0.9rem;
        }
        .upload-area {
            border: 3px dashed #ccc;
            border-radius: 10px;
            padding: 3rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            margin: 1.5rem 0;
        }
        .upload-area:hover {
            border-color: #667eea;
            background: #f0f3ff;
        }
        .upload-area input { display: none; }
        .preview-image {
            max-width: 100%;
            max-height: 300px;
            border-radius: 10px;
            margin: 1rem 0;
            display: none;
        }
        .analyze-btn {
            width: 100%;
            padding: 1rem;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1.2rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            margin: 1rem 0;
        }
        .analyze-btn:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        .analyze-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .result-card {
            background: linear-gradient(135deg, #667eea, #764ba2);
            border-radius: 15px;
            padding: 2rem;
            color: white;
            margin-top: 2rem;
        }
        .prediction {
            font-size: 2rem;
            font-weight: bold;
            margin: 1rem 0;
            text-transform: capitalize;
        }
        .confidence-bar {
            background: rgba(255,255,255,0.2);
            border-radius: 10px;
            height: 30px;
            margin: 1rem 0;
            overflow: hidden;
        }
        .confidence-fill {
            height: 100%;
            background: white;
            color: #667eea;
            line-height: 30px;
            padding-left: 1rem;
            transition: width 0.5s;
        }
        .prob-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }
        .prob-item {
            background: rgba(255,255,255,0.1);
            padding: 0.75rem;
            border-radius: 8px;
        }
        .trial-warning {
            background: #feebc8;
            color: #744210;
            padding: 1rem;
            border-radius: 10px;
            margin-top: 1rem;
            text-align: center;
        }
        .guidance-section {
            margin-top: 20px;
            padding-top: 20px;
            border-top: 2px solid rgba(255,255,255,0.2);
        }
        .safety-warning {
            background: rgba(255,0,0,0.2);
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 15px;
            border-left: 4px solid #ff4444;
        }
        .steps-list {
            margin-left: 20px;
            margin-bottom: 15px;
        }
        .steps-list li {
            margin-bottom: 8px;
            color: white;
        }
        .tools-container {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 10px;
        }
        .tool-tag {
            background: rgba(255,255,255,0.2);
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9rem;
        }
    </style>
</head>
<body>
    <nav class="navbar">
        <div class="logo">🔧 Trade AI Mentor</div>
        <div class="nav-links">
            <a href="{{ url_for('dashboard') }}">Dashboard</a>
            <a href="{{ url_for('analyze_page') }}">Analyze</a>
            <a href="{{ url_for('pricing') }}">Pricing</a>
            <a href="{{ url_for('logout') }}">Logout</a>
        </div>
    </nav>
    
    <div class="container">
        <div class="analyze-card">
            <h1 style="margin-bottom: 1.5rem;">🔍 New Analysis</h1>
            
            <div class="trade-selector">
                <button class="trade-btn construction" onclick="selectTrade('construction')">🏗️ Construction</button>
                <button class="trade-btn plumbing" onclick="selectTrade('plumbing')">🔧 Plumbing</button>
                <button class="trade-btn electrical" onclick="selectTrade('electrical')">⚡ Electrical</button>
            </div>
            
            <div id="modelInfo" class="model-info">
                <span id="tradeName">Select a trade to begin</span>
                <span id="tradeAccuracy" class="accuracy-badge"></span>
            </div>
            
            <div id="defectList" class="defect-tags"></div>
            
            {% if not current_user.is_subscribed() and current_user.free_trials_used < current_user.max_free_trials %}
                <div class="trial-warning">
                    <strong>Free Trial:</strong> {{ current_user.free_trials_used }}/{{ current_user.max_free_trials }} analyses remaining
                </div>
            {% endif %}
            
            <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                <input type="file" id="fileInput" accept="image/*" onchange="previewImage(event)">
                <p>📸 Click to upload or drag and drop</p>
                <p style="color: #999; font-size: 0.9em;">Supported: JPG, PNG (max 10MB)</p>
            </div>
            
            <img id="preview" class="preview-image">
            
            <button class="analyze-btn" onclick="analyze()" id="analyzeBtn" disabled>🔍 Analyze Image</button>
            
            <div id="result" style="display: none;">
                <div class="result-card">
                    <h3>Analysis Results</h3>
                    <div id="prediction" class="prediction"></div>
                    <div class="confidence-bar">
                        <div id="confidenceFill" class="confidence-fill" style="width: 0%">0%</div>
                    </div>
                    <div id="probabilities" class="prob-grid"></div>
                    
                    <!-- Repair Guidance Section -->
                    <div id="guidance-section" class="guidance-section" style="display: none;">
                        <h4 style="margin-bottom: 15px;">🔧 Repair Guidance</h4>
                        
                        <div id="safety-warning" class="safety-warning" style="display: none;">
                            <strong>⚠️ SAFETY WARNING</strong>
                            <p id="safety-text" style="margin-top: 5px;"></p>
                        </div>
                        
                        <div id="steps-section">
                            <h5 style="margin-bottom: 10px;">Step-by-Step Instructions:</h5>
                            <ol id="steps-list" class="steps-list"></ol>
                        </div>
                        
                        <div id="tools-section" style="margin-top: 15px;">
                            <h5 style="margin-bottom: 10px;">Tools Needed:</h5>
                            <div id="tools-list" class="tools-container"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let currentTrade = '';
        let currentImage = null;
        
        const accuracies = {{ accuracies|tojson }};
        const classNames = {{ CLASS_NAMES|tojson }};
        
        function selectTrade(trade) {
            currentTrade = trade;
            
            document.querySelectorAll('.trade-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            event.target.classList.add('active');
            
            document.getElementById('tradeName').textContent = trade.charAt(0).toUpperCase() + trade.slice(1);
            document.getElementById('tradeAccuracy').textContent = accuracies[trade] + '% Accuracy';
            
            const defectList = document.getElementById('defectList');
            defectList.innerHTML = '';
            classNames[trade].forEach(defect => {
                const tag = document.createElement('span');
                tag.className = 'defect-tag';
                tag.textContent = defect.replace(/_/g, ' ');
                defectList.appendChild(tag);
            });
            
            document.getElementById('analyzeBtn').disabled = !currentImage;
        }
        
        function previewImage(event) {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    const preview = document.getElementById('preview');
                    preview.src = e.target.result;
                    preview.style.display = 'block';
                    currentImage = file;
                    document.getElementById('analyzeBtn').disabled = !currentTrade;
                };
                reader.readAsDataURL(file);
            }
        }
        
async function analyze() {
    if (!currentTrade) {
        alert('Please select a trade first');
        return;
    }
    
    if (!currentImage) {
        alert('Please select an image');
        return;
    }
    
    const btn = document.getElementById('analyzeBtn');
    btn.disabled = true;
    btn.textContent = '🔍 Analyzing...';
    
    const formData = new FormData();
    formData.append('image', currentImage);
    formData.append('trade', currentTrade);
    
    console.log('Analyzing trade:', currentTrade);
    console.log('Image:', currentImage.name);
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        console.log('Response data:', data);
        
        if (data.error) {
            alert('Error: ' + data.error);
            console.error('Analysis error:', data.error);
        } else {
            // Display prediction
            document.getElementById('prediction').textContent = data.prediction.replace(/_/g, ' ');
            const confidencePercent = (data.confidence * 100).toFixed(1);
            document.getElementById('confidenceFill').style.width = confidencePercent + '%';
            document.getElementById('confidenceFill').textContent = confidencePercent + '%';
            
            // Display probabilities
            const probsDiv = document.getElementById('probabilities');
            probsDiv.innerHTML = '';
            
            for (const [className, prob] of Object.entries(data.probabilities)) {
                const probPercent = (prob * 100).toFixed(1);
                const item = document.createElement('div');
                item.className = 'prob-item';
                item.innerHTML = `
                    <strong>${className.replace(/_/g, ' ')}</strong><br>
                    ${probPercent}%
                    <div style="background: rgba(255,255,255,0.2); height: 5px; margin-top: 5px;">
                        <div style="background: white; width: ${probPercent}%; height: 100%;"></div>
                    </div>
                `;
                probsDiv.appendChild(item);
            }
            
            // Display repair guidance
            const guidanceSection = document.getElementById('guidance-section');
            console.log('Guidance data:', data.guidance);
            
            if (data.guidance) {
                guidanceSection.style.display = 'block';
                
                // Safety warning
                const safetyWarning = document.getElementById('safety-warning');
                const safetyText = document.getElementById('safety-text');
                
                if (data.guidance.safety && data.guidance.safety.length > 0) {
                    safetyText.textContent = data.guidance.safety;
                    safetyWarning.style.display = 'block';
                } else {
                    safetyWarning.style.display = 'none';
                }
                
                // Steps list
                const stepsList = document.getElementById('steps-list');
                stepsList.innerHTML = '';
                if (data.guidance.steps && data.guidance.steps.length > 0) {
                    data.guidance.steps.forEach(step => {
                        const li = document.createElement('li');
                        li.textContent = step;
                        stepsList.appendChild(li);
                    });
                }
                
                // Tools list
                const toolsList = document.getElementById('tools-list');
                toolsList.innerHTML = '';
                if (data.guidance.tools && data.guidance.tools.length > 0) {
                    data.guidance.tools.forEach(tool => {
                        const span = document.createElement('span');
                        span.className = 'tool-tag';
                        span.textContent = tool;
                        toolsList.appendChild(span);
                    });
                }
            } else {
                guidanceSection.style.display = 'none';
                console.warn('No guidance data received');
            }
            
            document.getElementById('result').style.display = 'block';
            
            // Update free trials display without alert
            if (data.free_trials_remaining !== undefined) {
                // Update the trial warning text on the page
                const trialWarning = document.querySelector('.trial-warning');
                if (trialWarning) {
                    trialWarning.innerHTML = `<strong>Free Trial:</strong> ${5 - data.free_trials_remaining}/5 analyses remaining`;
                }
                
                // Show a small notification instead of alert
                const notification = document.createElement('div');
                notification.style.position = 'fixed';
                notification.style.bottom = '20px';
                notification.style.right = '20px';
                notification.style.backgroundColor = '#48bb78';
                notification.style.color = 'white';
                notification.style.padding = '10px 20px';
                notification.style.borderRadius = '5px';
                notification.style.boxShadow = '0 2px 10px rgba(0,0,0,0.1)';
                notification.style.zIndex = '1000';
                notification.style.animation = 'slideIn 0.3s ease';
                notification.textContent = `✅ Free trials remaining: ${data.free_trials_remaining}`;
                document.body.appendChild(notification);
                
                // Remove notification after 3 seconds
                setTimeout(() => {
                    notification.remove();
                }, 3000);
            }
        }
    } catch (error) {
        console.error('Fetch error:', error);
        alert('Error analyzing image: ' + error);
    } finally {
        btn.disabled = false;
        btn.textContent = '🔍 Analyze Image';
    }
}
    </script>
</body>
</html>
'''
# ============================================
# MAIN
# ============================================
if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 TRADE AI MENTOR - WEB APP WITH SUBSCRIPTIONS")
    print("="*70)
    print("\n📊 Models loaded:")
    for trade in models_dict:
        status = "✅" if models_dict[trade] else "❌"
        print(f"   {status} {trade.capitalize()}: {MODEL_ACCURACIES[trade]}% accuracy")
    
    print("\n📱 Features:")
    print("   ✅ User authentication")
    print("   ✅ 5 free trial analyses")
    print("   ✅ Stripe subscriptions")
    print("   ✅ Dashboard with history")
    print("   ✅ Beautiful modern UI")
    print("   ✅ Repair guidance with steps and tools")
    
    print("\n🌐 Open http://localhost:5000 in your browser")
    print("="*70)
    app.run(host='0.0.0.0', port=5000, debug=True)
