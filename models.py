from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
import json
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class Admin(db.Model):
    __tablename__ = 'admins'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class AccessCode(db.Model):
    __tablename__ = 'access_codes'
    
    code = db.Column(db.String(10), primary_key=True)
    status = db.Column(db.String(10), default='active')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    first_login_at = db.Column(db.DateTime, nullable=True)
    last_login_at = db.Column(db.DateTime, nullable=True)
    login_attempts = db.Column(db.Integer, default=0)
    
    # mode flags - active or passive for each feature
    report_version = db.Column(db.String(20), default='practice')
    localize_mode = db.Column(db.String(10), default='active')
    report_mode = db.Column(db.String(10), default='active')
    
    # test completion tracking
    took_localize_pre = db.Column(db.Boolean, default=False)
    took_localize_post = db.Column(db.Boolean, default=False)
    took_report_pre = db.Column(db.Boolean, default=False)
    took_report_post = db.Column(db.Boolean, default=False)
    localize_cases_completed = db.Column(db.Integer, default=0)
    report_cases_completed = db.Column(db.Integer, default=0)
    
    activities = db.relationship('ActivityLog', backref='access_code', lazy=True)

    def to_dict(self):
        case_activities = ActivityLog.query.filter_by(
            access_code_id=self.code,
            activity_type='case_completion'
        ).count()
        
        correct_cases = ActivityLog.query.filter_by(
            access_code_id=self.code,
            activity_type='case_completion',
            is_correct=True
        ).count()
        
        return {
            'code': self.code,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'first_login_at': self.first_login_at.isoformat() if self.first_login_at else None,
            'last_login_at': self.last_login_at.isoformat() if self.last_login_at else None,
            'login_attempts': self.login_attempts,
            'total_cases': case_activities,
            'correct_cases': correct_cases,
            'accuracy': f"{(correct_cases / case_activities * 100):.1f}%" if case_activities > 0 else "N/A",
            'localize_mode': getattr(self, 'localize_mode', 'active'),
            'report_mode': getattr(self, 'report_mode', 'active'),
            'report_version': getattr(self, 'report_version', 'practice')
        }

class BoundingBoxLog(db.Model):
    __tablename__ = 'bounding_box_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    activity_log_id = db.Column(db.Integer, db.ForeignKey('activity_logs.id'), nullable=False)
    label = db.Column(db.String(50), nullable=False)
    x1 = db.Column(db.Float, nullable=False)
    y1 = db.Column(db.Float, nullable=False)
    x2 = db.Column(db.Float, nullable=False)
    y2 = db.Column(db.Float, nullable=False)
    is_ground_truth = db.Column(db.Boolean, nullable=False)
    confidence_score = db.Column(db.Float, nullable=True)
    
    def to_dict(self):
        return {
            'label': self.label,
            'coordinates': [self.x1, self.y1, self.x2, self.y2],
            'is_ground_truth': self.is_ground_truth,
            'confidence_score': self.confidence_score
        }

class ActivityLog(db.Model):
    __tablename__ = 'activity_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    access_code_id = db.Column(db.String(10), db.ForeignKey('access_codes.code'), nullable=False)
    activity_type = db.Column(db.String(50), nullable=False)
    case_id = db.Column(db.String(50), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    is_correct = db.Column(db.Boolean, nullable=True)
    activity_metadata = db.Column(db.Text, nullable=True)

    bounding_boxes = db.relationship('BoundingBoxLog', backref='activity', lazy=True)

    def set_metadata(self, data):
        self.activity_metadata = json.dumps(data)

    def get_metadata(self):
        metadata = {}
        if self.activity_metadata:
            try:
                metadata = json.loads(self.activity_metadata)
            except Exception:
                metadata = {'raw': self.activity_metadata}
        if self.bounding_boxes:
            metadata['bounding_boxes'] = {
                'ground_truth': [box.to_dict() for box in self.bounding_boxes if box.is_ground_truth],
                'user_submission': [box.to_dict() for box in self.bounding_boxes if not box.is_ground_truth]
            }
        return metadata