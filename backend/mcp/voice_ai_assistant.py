"""
Voice AI Assistant - 40by6
Natural language interface for MCP Stack with advanced AI capabilities
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForQuestionAnswering,
    AutoModelForCausalLM, AutoModelForSequenceClassification,
    Wav2Vec2ForCTC, Wav2Vec2Processor, BlenderbotForConditionalGeneration,
    T5ForConditionalGeneration, GPT2LMHeadModel, BertForQuestionAnswering
)
import whisper
import sounddevice as sd
import soundfile as sf
import pyttsx3
from gtts import gTTS
import speech_recognition as sr
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy
import langchain
from langchain.llms import OpenAI, Anthropic, Cohere
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma, Pinecone, Weaviate
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
import openai
import anthropic
import cohere
from sentence_transformers import SentenceTransformer
import faiss
import redis
from sqlalchemy import create_engine, Column, String, DateTime, Integer, Float, JSON, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import asyncpg
from prometheus_client import Counter, Histogram, Gauge
import websockets
import aiohttp
from pydub import AudioSegment
from scipy.io import wavfile
import librosa
import webrtcvad
import pyaudio
import wave
import os
import tempfile
import hashlib
from concurrent.futures import ThreadPoolExecutor
import ray
from typing_extensions import Protocol

logger = logging.getLogger(__name__)

# Download required NLTK data
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_lg")
except:
    os.system("python -m spacy download en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")

# Metrics
voice_commands = Counter('voice_commands_total', 'Total voice commands processed', ['command_type', 'status'])
response_time = Histogram('voice_response_duration_seconds', 'Voice response time')
active_sessions = Gauge('voice_active_sessions', 'Currently active voice sessions')
sentiment_scores = Histogram('voice_sentiment_scores', 'Sentiment analysis scores', ['sentiment'])

Base = declarative_base()


class VoiceCommandType(Enum):
    """Types of voice commands"""
    QUERY = "query"  # Information retrieval
    ACTION = "action"  # Execute action
    CONVERSATION = "conversation"  # General chat
    SYSTEM = "system"  # System control
    ANALYSIS = "analysis"  # Data analysis
    REPORT = "report"  # Generate report
    ALERT = "alert"  # Create alert
    CONFIG = "config"  # Configuration change
    NAVIGATION = "navigation"  # UI navigation
    HELP = "help"  # Help request


class EmotionalTone(Enum):
    """Emotional tones for responses"""
    NEUTRAL = "neutral"
    FRIENDLY = "friendly"
    PROFESSIONAL = "professional"
    URGENT = "urgent"
    ENCOURAGING = "encouraging"
    SERIOUS = "serious"
    CASUAL = "casual"
    EXCITED = "excited"


class Language(Enum):
    """Supported languages"""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    CHINESE = "zh"
    JAPANESE = "ja"
    KOREAN = "ko"
    ARABIC = "ar"
    HINDI = "hi"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    ITALIAN = "it"


@dataclass
class VoiceSession:
    """Voice interaction session"""
    id: str
    user_id: str
    language: Language = Language.ENGLISH
    emotional_tone: EmotionalTone = EmotionalTone.FRIENDLY
    context: Dict[str, Any] = field(default_factory=dict)
    history: List[Tuple[str, str]] = field(default_factory=list)  # (user_input, assistant_response)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_interaction: datetime = field(default_factory=datetime.utcnow)
    is_active: bool = True
    wake_word_detected: bool = False
    continuous_listening: bool = False
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    skill_modules: List[str] = field(default_factory=list)


@dataclass
class VoiceCommand:
    """Parsed voice command"""
    raw_text: str
    command_type: VoiceCommandType
    intent: str
    entities: Dict[str, Any]
    confidence: float
    sentiment: float
    emotion: str
    language: Language
    context_required: bool = False
    requires_confirmation: bool = False
    priority: int = 5  # 1-10, higher is more important


@dataclass
class VoiceResponse:
    """Voice assistant response"""
    text: str
    audio_data: Optional[bytes] = None
    emotion: EmotionalTone = EmotionalTone.NEUTRAL
    actions: List[Dict[str, Any]] = field(default_factory=list)
    visualizations: List[Dict[str, Any]] = field(default_factory=list)
    follow_up_questions: List[str] = field(default_factory=list)
    confidence: float = 1.0
    source_references: List[str] = field(default_factory=list)


class ConversationHistory(Base):
    """Store conversation history"""
    __tablename__ = 'voice_conversation_history'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String(50), index=True)
    user_id = Column(String(50), index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    user_input = Column(Text)
    assistant_response = Column(Text)
    command_type = Column(String(50))
    intent = Column(String(100))
    entities = Column(JSON)
    sentiment = Column(Float)
    emotion = Column(String(50))
    audio_duration = Column(Float)
    response_time = Column(Float)
    success = Column(Boolean, default=True)
    feedback = Column(JSON)


class VoiceSkill(Protocol):
    """Protocol for voice skills"""
    
    name: str
    description: str
    keywords: List[str]
    
    async def can_handle(self, command: VoiceCommand) -> bool:
        """Check if skill can handle command"""
        ...
    
    async def handle(self, command: VoiceCommand, session: VoiceSession) -> VoiceResponse:
        """Handle the command"""
        ...


class NLUEngine:
    """Natural Language Understanding Engine"""
    
    def __init__(self):
        self.intent_classifier = None
        self.entity_extractor = None
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self._load_models()
    
    def _load_models(self):
        """Load NLU models"""
        try:
            # Load intent classification model
            self.intent_classifier = AutoModelForSequenceClassification.from_pretrained(
                "distilbert-base-uncased-finetuned-sst-2-english"
            )
            self.intent_tokenizer = AutoTokenizer.from_pretrained(
                "distilbert-base-uncased-finetuned-sst-2-english"
            )
        except Exception as e:
            logger.warning(f"Could not load intent classifier: {e}")
    
    async def understand(self, text: str, language: Language = Language.ENGLISH) -> VoiceCommand:
        """Understand natural language input"""
        
        # Clean text
        text = text.strip().lower()
        
        # Detect command type and intent
        command_type, intent = await self._classify_intent(text)
        
        # Extract entities
        entities = await self._extract_entities(text)
        
        # Sentiment analysis
        sentiment = self._analyze_sentiment(text)
        
        # Emotion detection
        emotion = self._detect_emotion(text)
        
        # Confidence scoring
        confidence = self._calculate_confidence(text, intent, entities)
        
        # Check if confirmation needed
        requires_confirmation = self._needs_confirmation(command_type, entities)
        
        return VoiceCommand(
            raw_text=text,
            command_type=command_type,
            intent=intent,
            entities=entities,
            confidence=confidence,
            sentiment=sentiment,
            emotion=emotion,
            language=language,
            requires_confirmation=requires_confirmation
        )
    
    async def _classify_intent(self, text: str) -> Tuple[VoiceCommandType, str]:
        """Classify intent from text"""
        
        # Rule-based classification for common patterns
        text_lower = text.lower()
        
        # Queries
        if any(word in text_lower for word in ['what', 'when', 'where', 'who', 'how', 'show me', 'tell me']):
            intent = self._get_query_intent(text_lower)
            return VoiceCommandType.QUERY, intent
        
        # Actions
        if any(word in text_lower for word in ['start', 'stop', 'run', 'execute', 'deploy', 'create', 'delete']):
            intent = self._get_action_intent(text_lower)
            return VoiceCommandType.ACTION, intent
        
        # Analysis
        if any(word in text_lower for word in ['analyze', 'compare', 'predict', 'forecast', 'trend']):
            return VoiceCommandType.ANALYSIS, "data_analysis"
        
        # Reports
        if any(word in text_lower for word in ['report', 'summary', 'export', 'generate']):
            return VoiceCommandType.REPORT, "generate_report"
        
        # System
        if any(word in text_lower for word in ['system', 'status', 'health', 'performance']):
            return VoiceCommandType.SYSTEM, "system_status"
        
        # Help
        if any(word in text_lower for word in ['help', 'assist', 'guide', 'tutorial']):
            return VoiceCommandType.HELP, "help_request"
        
        # Default to conversation
        return VoiceCommandType.CONVERSATION, "general_chat"
    
    def _get_query_intent(self, text: str) -> str:
        """Get specific query intent"""
        if 'scraper' in text:
            return "query_scrapers"
        elif 'data' in text:
            return "query_data"
        elif 'status' in text:
            return "query_status"
        elif 'metric' in text or 'performance' in text:
            return "query_metrics"
        else:
            return "general_query"
    
    def _get_action_intent(self, text: str) -> str:
        """Get specific action intent"""
        if 'scraper' in text:
            if 'start' in text:
                return "start_scraper"
            elif 'stop' in text:
                return "stop_scraper"
            else:
                return "manage_scraper"
        elif 'deploy' in text:
            return "deploy_system"
        elif 'backup' in text:
            return "create_backup"
        else:
            return "general_action"
    
    async def _extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract entities from text"""
        
        doc = nlp(text)
        entities = {}
        
        # Named entities
        for ent in doc.ents:
            entities[ent.label_] = entities.get(ent.label_, [])
            entities[ent.label_].append(ent.text)
        
        # Time expressions
        time_expressions = self._extract_time_expressions(text)
        if time_expressions:
            entities['TIME'] = time_expressions
        
        # Numbers
        numbers = [token.text for token in doc if token.pos_ == 'NUM']
        if numbers:
            entities['NUMBER'] = numbers
        
        # Custom entities for MCP
        if 'scraper' in text:
            # Extract scraper names or IDs
            scraper_patterns = self._extract_scraper_references(text)
            if scraper_patterns:
                entities['SCRAPER'] = scraper_patterns
        
        return entities
    
    def _extract_time_expressions(self, text: str) -> List[str]:
        """Extract time expressions"""
        time_patterns = [
            'today', 'yesterday', 'tomorrow',
            'last hour', 'last day', 'last week', 'last month',
            'next hour', 'next day', 'next week', 'next month',
            'now', 'currently', 'recently'
        ]
        
        found = []
        for pattern in time_patterns:
            if pattern in text.lower():
                found.append(pattern)
        
        return found
    
    def _extract_scraper_references(self, text: str) -> List[str]:
        """Extract scraper references"""
        # This would be enhanced with actual scraper name/ID recognition
        words = text.split()
        scrapers = []
        
        for i, word in enumerate(words):
            if word.lower() == 'scraper' and i + 1 < len(words):
                scrapers.append(words[i + 1])
        
        return scrapers
    
    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text"""
        scores = self.sentiment_analyzer.polarity_scores(text)
        return scores['compound']
    
    def _detect_emotion(self, text: str) -> str:
        """Detect emotion from text"""
        scores = self.sentiment_analyzer.polarity_scores(text)
        
        if scores['compound'] >= 0.5:
            return "positive"
        elif scores['compound'] <= -0.5:
            return "negative"
        elif scores['neg'] > 0.5:
            return "angry"
        elif scores['pos'] > 0.5:
            return "happy"
        else:
            return "neutral"
    
    def _calculate_confidence(self, text: str, intent: str, entities: Dict[str, Any]) -> float:
        """Calculate confidence score"""
        confidence = 0.7  # Base confidence
        
        # Adjust based on text length
        if len(text.split()) < 3:
            confidence -= 0.2
        elif len(text.split()) > 20:
            confidence -= 0.1
        
        # Adjust based on entities found
        if entities:
            confidence += 0.1
        
        # Ensure in valid range
        return max(0.0, min(1.0, confidence))
    
    def _needs_confirmation(self, command_type: VoiceCommandType, entities: Dict[str, Any]) -> bool:
        """Check if command needs confirmation"""
        
        # Actions that modify state need confirmation
        dangerous_actions = [
            VoiceCommandType.ACTION,
            VoiceCommandType.CONFIG
        ]
        
        if command_type in dangerous_actions:
            # Don't need confirmation for safe queries
            safe_keywords = ['status', 'list', 'show', 'get']
            return not any(keyword in entities.get('intent', '') for keyword in safe_keywords)
        
        return False


class SpeechRecognitionEngine:
    """Advanced speech recognition with multiple engines"""
    
    def __init__(self):
        self.whisper_model = None
        self.wav2vec_model = None
        self.recognizer = sr.Recognizer()
        self.vad = webrtcvad.Vad(3)  # Aggressiveness level 3
        self._load_models()
    
    def _load_models(self):
        """Load speech recognition models"""
        try:
            # Load Whisper
            self.whisper_model = whisper.load_model("base")
            
            # Load Wav2Vec2
            self.wav2vec_processor = Wav2Vec2Processor.from_pretrained(
                "facebook/wav2vec2-base-960h"
            )
            self.wav2vec_model = Wav2Vec2ForCTC.from_pretrained(
                "facebook/wav2vec2-base-960h"
            )
        except Exception as e:
            logger.warning(f"Could not load speech models: {e}")
    
    async def recognize_speech(
        self,
        audio_data: Union[bytes, np.ndarray],
        language: Language = Language.ENGLISH,
        use_enhanced: bool = True
    ) -> Tuple[str, float]:
        """Recognize speech from audio data"""
        
        if use_enhanced and self.whisper_model:
            return await self._recognize_whisper(audio_data, language)
        else:
            return await self._recognize_standard(audio_data, language)
    
    async def _recognize_whisper(
        self,
        audio_data: Union[bytes, np.ndarray],
        language: Language
    ) -> Tuple[str, float]:
        """Use Whisper for recognition"""
        
        # Convert audio to numpy array if needed
        if isinstance(audio_data, bytes):
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            audio_array = audio_data
        
        # Transcribe
        result = self.whisper_model.transcribe(
            audio_array,
            language=language.value,
            task="transcribe"
        )
        
        text = result["text"]
        
        # Calculate confidence from log probability
        confidence = np.exp(result.get("avg_logprob", -0.5))
        
        return text.strip(), confidence
    
    async def _recognize_standard(
        self,
        audio_data: bytes,
        language: Language
    ) -> Tuple[str, float]:
        """Use standard speech recognition"""
        
        try:
            # Convert to AudioData
            audio = sr.AudioData(audio_data, 16000, 2)
            
            # Try multiple recognition engines
            text = ""
            confidence = 0.5
            
            try:
                # Try Google
                text = self.recognizer.recognize_google(
                    audio,
                    language=language.value
                )
                confidence = 0.8
            except:
                try:
                    # Try Sphinx as fallback
                    text = self.recognizer.recognize_sphinx(audio)
                    confidence = 0.6
                except:
                    pass
            
            return text, confidence
            
        except Exception as e:
            logger.error(f"Speech recognition failed: {e}")
            return "", 0.0
    
    async def detect_wake_word(
        self,
        audio_stream: bytes,
        wake_words: List[str] = ["hey mcp", "okay mcp", "mcp"]
    ) -> bool:
        """Detect wake word in audio stream"""
        
        # Convert to text
        text, confidence = await self.recognize_speech(audio_stream, use_enhanced=False)
        
        if confidence < 0.5:
            return False
        
        text_lower = text.lower()
        return any(wake_word in text_lower for wake_word in wake_words)
    
    def is_speech(self, audio_chunk: bytes, sample_rate: int = 16000) -> bool:
        """Check if audio chunk contains speech using VAD"""
        
        # Ensure chunk is correct size for VAD (10, 20, or 30 ms)
        chunk_duration_ms = 30
        chunk_size = int(sample_rate * chunk_duration_ms / 1000) * 2  # 2 bytes per sample
        
        if len(audio_chunk) < chunk_size:
            return False
        
        # Process in chunks
        for i in range(0, len(audio_chunk) - chunk_size + 1, chunk_size):
            chunk = audio_chunk[i:i + chunk_size]
            if self.vad.is_speech(chunk, sample_rate):
                return True
        
        return False


class TextToSpeechEngine:
    """Advanced text-to-speech with multiple voices and emotions"""
    
    def __init__(self):
        self.tts_engine = pyttsx3.init()
        self.voices = self.tts_engine.getProperty('voices')
        self.voice_models = {}
        self._setup_voices()
    
    def _setup_voices(self):
        """Setup available voices"""
        # Configure pyttsx3
        self.tts_engine.setProperty('rate', 175)  # Speech rate
        self.tts_engine.setProperty('volume', 0.9)  # Volume
        
        # Select appropriate voice
        for voice in self.voices:
            if 'female' in voice.name.lower():
                self.tts_engine.setProperty('voice', voice.id)
                break
    
    async def synthesize_speech(
        self,
        text: str,
        language: Language = Language.ENGLISH,
        emotion: EmotionalTone = EmotionalTone.NEUTRAL,
        voice_id: Optional[str] = None,
        speed: float = 1.0,
        pitch: float = 1.0
    ) -> bytes:
        """Synthesize speech from text"""
        
        # Apply emotional tone to text
        text = self._apply_emotional_tone(text, emotion)
        
        # Generate speech
        if language == Language.ENGLISH:
            return await self._synthesize_pyttsx3(text, speed)
        else:
            return await self._synthesize_gtts(text, language)
    
    def _apply_emotional_tone(self, text: str, emotion: EmotionalTone) -> str:
        """Apply emotional tone to text"""
        
        if emotion == EmotionalTone.FRIENDLY:
            # Add friendly markers
            if not text.endswith('!') and not text.endswith('?'):
                text = text.rstrip('.') + '!'
        
        elif emotion == EmotionalTone.URGENT:
            # Emphasize urgency
            text = f"ATTENTION: {text}"
        
        elif emotion == EmotionalTone.ENCOURAGING:
            # Add encouraging prefix
            encouragements = ["Great! ", "Excellent! ", "Wonderful! "]
            text = np.random.choice(encouragements) + text
        
        return text
    
    async def _synthesize_pyttsx3(self, text: str, speed: float) -> bytes:
        """Synthesize using pyttsx3"""
        
        # Adjust speed
        current_rate = self.tts_engine.getProperty('rate')
        self.tts_engine.setProperty('rate', int(current_rate * speed))
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        self.tts_engine.save_to_file(text, tmp_path)
        self.tts_engine.runAndWait()
        
        # Read audio data
        with open(tmp_path, 'rb') as f:
            audio_data = f.read()
        
        # Cleanup
        os.unlink(tmp_path)
        
        return audio_data
    
    async def _synthesize_gtts(self, text: str, language: Language) -> bytes:
        """Synthesize using gTTS for non-English"""
        
        tts = gTTS(text=text, lang=language.value, slow=False)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        tts.save(tmp_path)
        
        # Convert MP3 to WAV
        audio = AudioSegment.from_mp3(tmp_path)
        wav_path = tmp_path.replace('.mp3', '.wav')
        audio.export(wav_path, format='wav')
        
        # Read WAV data
        with open(wav_path, 'rb') as f:
            audio_data = f.read()
        
        # Cleanup
        os.unlink(tmp_path)
        os.unlink(wav_path)
        
        return audio_data


class ConversationManager:
    """Manage multi-turn conversations with context"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.sessions: Dict[str, VoiceSession] = {}
        self.memory_store = ConversationBufferWindowMemory(k=10)
        self.context_embeddings = {}
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Setup database
        self.engine = create_engine(database_url)
        self.Session = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)
    
    async def create_session(
        self,
        user_id: str,
        language: Language = Language.ENGLISH
    ) -> VoiceSession:
        """Create new conversation session"""
        
        session_id = f"voice_{user_id}_{datetime.utcnow().timestamp()}"
        
        session = VoiceSession(
            id=session_id,
            user_id=user_id,
            language=language
        )
        
        self.sessions[session_id] = session
        active_sessions.inc()
        
        return session
    
    async def update_context(
        self,
        session: VoiceSession,
        user_input: str,
        assistant_response: str
    ):
        """Update conversation context"""
        
        # Add to history
        session.history.append((user_input, assistant_response))
        
        # Limit history size
        if len(session.history) > 20:
            session.history = session.history[-20:]
        
        # Update last interaction
        session.last_interaction = datetime.utcnow()
        
        # Store in database
        db_session = self.Session()
        try:
            history_entry = ConversationHistory(
                session_id=session.id,
                user_id=session.user_id,
                user_input=user_input,
                assistant_response=assistant_response,
                timestamp=datetime.utcnow()
            )
            db_session.add(history_entry)
            db_session.commit()
        finally:
            db_session.close()
        
        # Update embeddings for context retrieval
        combined_text = f"{user_input} {assistant_response}"
        embedding = self.embedding_model.encode(combined_text)
        self.context_embeddings[session.id] = embedding
    
    async def get_relevant_context(
        self,
        session: VoiceSession,
        query: str,
        k: int = 5
    ) -> List[Tuple[str, str]]:
        """Get relevant context from history"""
        
        # Get query embedding
        query_embedding = self.embedding_model.encode(query)
        
        # Calculate similarities with history
        similarities = []
        for i, (user_input, response) in enumerate(session.history):
            hist_embedding = self.embedding_model.encode(f"{user_input} {response}")
            similarity = np.dot(query_embedding, hist_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(hist_embedding)
            )
            similarities.append((similarity, i))
        
        # Get top k similar exchanges
        similarities.sort(reverse=True)
        relevant_context = []
        
        for similarity, idx in similarities[:k]:
            if similarity > 0.5:  # Threshold for relevance
                relevant_context.append(session.history[idx])
        
        return relevant_context
    
    async def end_session(self, session_id: str):
        """End conversation session"""
        
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.is_active = False
            
            # Clean up
            del self.sessions[session_id]
            if session_id in self.context_embeddings:
                del self.context_embeddings[session_id]
            
            active_sessions.dec()


class ResponseGenerator:
    """Generate natural language responses"""
    
    def __init__(self):
        self.templates = self._load_templates()
        self.llm = None
        self._setup_llm()
    
    def _setup_llm(self):
        """Setup language model"""
        try:
            # Try to use local model first
            self.model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
            self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        except:
            logger.warning("Could not load local language model")
    
    def _load_templates(self) -> Dict[str, List[str]]:
        """Load response templates"""
        return {
            'greeting': [
                "Hello! How can I assist you today?",
                "Hi there! What can I help you with?",
                "Greetings! I'm here to help with your MCP Stack needs."
            ],
            'query_scrapers': [
                "I found {count} scrapers matching your criteria. {details}",
                "Here's the scraper information you requested: {details}",
                "The {scraper_name} scraper is currently {status}. {additional_info}"
            ],
            'action_success': [
                "I've successfully {action} the {target}.",
                "Done! The {target} has been {action}.",
                "{action} completed successfully for {target}."
            ],
            'action_failure': [
                "I'm sorry, I couldn't {action} the {target}. {reason}",
                "There was an issue with {action}. {reason}",
                "Unable to complete {action} for {target}. {reason}"
            ],
            'clarification': [
                "Could you please clarify what you mean by '{term}'?",
                "I need more information about {aspect}. Can you elaborate?",
                "To help you better, could you specify {detail}?"
            ],
            'analysis_result': [
                "Based on my analysis, {finding}. {recommendation}",
                "The data shows {trend}. {insight}",
                "I've analyzed {data_type} and found {result}."
            ],
            'error': [
                "I apologize, but I encountered an error: {error_message}",
                "Something went wrong. {error_message}",
                "I'm having trouble with that request. {error_message}"
            ]
        }
    
    async def generate_response(
        self,
        command: VoiceCommand,
        context: Dict[str, Any],
        tone: EmotionalTone = EmotionalTone.FRIENDLY
    ) -> VoiceResponse:
        """Generate response for command"""
        
        # Get base response based on intent
        if command.intent == "greeting":
            text = np.random.choice(self.templates['greeting'])
        
        elif command.intent.startswith("query_"):
            text = await self._generate_query_response(command, context)
        
        elif command.intent.startswith("action_"):
            text = await self._generate_action_response(command, context)
        
        elif command.intent == "analysis":
            text = await self._generate_analysis_response(command, context)
        
        else:
            text = await self._generate_general_response(command, context)
        
        # Apply tone adjustments
        text = self._apply_tone(text, tone)
        
        # Generate follow-up questions if needed
        follow_ups = self._generate_follow_ups(command, context)
        
        # Create response
        response = VoiceResponse(
            text=text,
            emotion=tone,
            follow_up_questions=follow_ups,
            confidence=command.confidence
        )
        
        # Add actions if applicable
        if command.command_type == VoiceCommandType.ACTION:
            response.actions = self._generate_actions(command, context)
        
        # Add visualizations if applicable
        if command.command_type in [VoiceCommandType.ANALYSIS, VoiceCommandType.REPORT]:
            response.visualizations = self._generate_visualizations(command, context)
        
        return response
    
    async def _generate_query_response(
        self,
        command: VoiceCommand,
        context: Dict[str, Any]
    ) -> str:
        """Generate response for queries"""
        
        template_key = command.intent.replace('query_', 'query_')
        if template_key in self.templates:
            template = np.random.choice(self.templates[template_key])
            
            # Fill in template with context
            response = template.format(**context)
        else:
            # Generate using LLM if available
            if self.model:
                prompt = f"Answer this question about MCP Stack: {command.raw_text}"
                response = await self._generate_with_llm(prompt)
            else:
                response = f"Here's what I found: {context.get('result', 'No data available')}"
        
        return response
    
    async def _generate_action_response(
        self,
        command: VoiceCommand,
        context: Dict[str, Any]
    ) -> str:
        """Generate response for actions"""
        
        success = context.get('success', False)
        template_key = 'action_success' if success else 'action_failure'
        
        template = np.random.choice(self.templates[template_key])
        
        response = template.format(
            action=context.get('action', 'process'),
            target=context.get('target', 'request'),
            reason=context.get('reason', 'Unknown error'),
            **context
        )
        
        return response
    
    async def _generate_analysis_response(
        self,
        command: VoiceCommand,
        context: Dict[str, Any]
    ) -> str:
        """Generate response for analysis"""
        
        template = np.random.choice(self.templates['analysis_result'])
        
        response = template.format(
            finding=context.get('finding', 'No significant patterns found'),
            recommendation=context.get('recommendation', ''),
            trend=context.get('trend', 'stable'),
            insight=context.get('insight', ''),
            data_type=context.get('data_type', 'the data'),
            result=context.get('result', 'inconclusive results')
        )
        
        return response
    
    async def _generate_general_response(
        self,
        command: VoiceCommand,
        context: Dict[str, Any]
    ) -> str:
        """Generate general response"""
        
        if self.model:
            return await self._generate_with_llm(command.raw_text)
        else:
            return "I understand you're asking about " + command.raw_text + ". Let me help you with that."
    
    async def _generate_with_llm(self, prompt: str) -> str:
        """Generate response using LLM"""
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=150,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return "I'm having trouble generating a response. Please try rephrasing your question."
    
    def _apply_tone(self, text: str, tone: EmotionalTone) -> str:
        """Apply emotional tone to text"""
        
        if tone == EmotionalTone.FRIENDLY:
            if not any(text.endswith(p) for p in ['.', '!', '?']):
                text += '!'
            
            # Add friendly phrases
            friendly_additions = [
                "I'd be happy to help! ",
                "Great question! ",
                "Thanks for asking! "
            ]
            if np.random.random() < 0.3:
                text = np.random.choice(friendly_additions) + text
        
        elif tone == EmotionalTone.PROFESSIONAL:
            # Remove exclamation marks
            text = text.replace('!', '.')
            
            # Add professional prefix
            if np.random.random() < 0.2:
                text = "As per your request, " + text
        
        elif tone == EmotionalTone.URGENT:
            text = "⚠️ " + text.upper()
        
        return text
    
    def _generate_follow_ups(
        self,
        command: VoiceCommand,
        context: Dict[str, Any]
    ) -> List[str]:
        """Generate follow-up questions"""
        
        follow_ups = []
        
        if command.confidence < 0.7:
            follow_ups.append("Was this what you were looking for?")
        
        if command.command_type == VoiceCommandType.QUERY:
            follow_ups.extend([
                "Would you like more details?",
                "Should I show you related information?"
            ])
        
        elif command.command_type == VoiceCommandType.ACTION:
            follow_ups.extend([
                "Would you like me to perform any related actions?",
                "Should I monitor this for you?"
            ])
        
        return follow_ups[:2]  # Limit to 2 follow-ups
    
    def _generate_actions(
        self,
        command: VoiceCommand,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate executable actions"""
        
        actions = []
        
        if 'scraper' in command.entities:
            actions.append({
                'type': 'manage_scraper',
                'scraper_id': command.entities.get('SCRAPER', ['unknown'])[0],
                'operation': command.intent
            })
        
        if 'report' in command.intent:
            actions.append({
                'type': 'generate_report',
                'report_type': context.get('report_type', 'summary'),
                'time_range': command.entities.get('TIME', ['last day'])[0]
            })
        
        return actions
    
    def _generate_visualizations(
        self,
        command: VoiceCommand,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate visualization recommendations"""
        
        visualizations = []
        
        if 'trend' in command.raw_text or 'time' in command.raw_text:
            visualizations.append({
                'type': 'line_chart',
                'data': 'time_series',
                'title': 'Trend Analysis'
            })
        
        if 'compare' in command.raw_text:
            visualizations.append({
                'type': 'bar_chart',
                'data': 'comparison',
                'title': 'Comparison View'
            })
        
        if 'distribution' in command.raw_text:
            visualizations.append({
                'type': 'pie_chart',
                'data': 'distribution',
                'title': 'Distribution Analysis'
            })
        
        return visualizations


class SkillsManager:
    """Manage voice assistant skills"""
    
    def __init__(self):
        self.skills: Dict[str, VoiceSkill] = {}
        self._load_builtin_skills()
    
    def _load_builtin_skills(self):
        """Load built-in skills"""
        
        # System Status Skill
        self.register_skill(SystemStatusSkill())
        
        # Scraper Management Skill
        self.register_skill(ScraperManagementSkill())
        
        # Analytics Skill
        self.register_skill(AnalyticsSkill())
        
        # Help Skill
        self.register_skill(HelpSkill())
    
    def register_skill(self, skill: VoiceSkill):
        """Register a skill"""
        self.skills[skill.name] = skill
        logger.info(f"Registered voice skill: {skill.name}")
    
    async def find_skill(self, command: VoiceCommand) -> Optional[VoiceSkill]:
        """Find appropriate skill for command"""
        
        # Check each skill
        for skill in self.skills.values():
            if await skill.can_handle(command):
                return skill
        
        return None


class SystemStatusSkill:
    """Skill for system status queries"""
    
    name = "system_status"
    description = "Query and report system status"
    keywords = ["status", "health", "performance", "system"]
    
    async def can_handle(self, command: VoiceCommand) -> bool:
        """Check if can handle command"""
        return command.intent in ["system_status", "query_status", "query_metrics"]
    
    async def handle(
        self,
        command: VoiceCommand,
        session: VoiceSession
    ) -> VoiceResponse:
        """Handle system status command"""
        
        # Fetch system status (mock data for demo)
        status = {
            'overall': 'healthy',
            'scrapers_active': 127,
            'success_rate': 98.5,
            'cpu_usage': 42,
            'memory_usage': 67
        }
        
        # Generate response text
        text = f"""System status is {status['overall']}. 
        We have {status['scrapers_active']} active scrapers with a {status['success_rate']}% success rate.
        CPU usage is at {status['cpu_usage']}% and memory usage is at {status['memory_usage']}%."""
        
        return VoiceResponse(
            text=text,
            emotion=EmotionalTone.PROFESSIONAL,
            visualizations=[
                {
                    'type': 'gauge',
                    'data': status,
                    'title': 'System Health'
                }
            ]
        )


class ScraperManagementSkill:
    """Skill for scraper management"""
    
    name = "scraper_management"
    description = "Manage scrapers via voice commands"
    keywords = ["scraper", "start", "stop", "run", "schedule"]
    
    async def can_handle(self, command: VoiceCommand) -> bool:
        """Check if can handle command"""
        return 'scraper' in command.raw_text.lower()
    
    async def handle(
        self,
        command: VoiceCommand,
        session: VoiceSession
    ) -> VoiceResponse:
        """Handle scraper command"""
        
        # Parse scraper action
        if 'start' in command.raw_text:
            action = 'started'
        elif 'stop' in command.raw_text:
            action = 'stopped'
        else:
            action = 'processed'
        
        scraper_name = command.entities.get('SCRAPER', ['the scraper'])[0]
        
        text = f"I've {action} {scraper_name} successfully."
        
        return VoiceResponse(
            text=text,
            emotion=EmotionalTone.FRIENDLY,
            actions=[{
                'type': 'scraper_action',
                'action': action,
                'target': scraper_name
            }]
        )


class AnalyticsSkill:
    """Skill for analytics and reporting"""
    
    name = "analytics"
    description = "Provide analytics and insights"
    keywords = ["analyze", "report", "trend", "insight", "data"]
    
    async def can_handle(self, command: VoiceCommand) -> bool:
        """Check if can handle command"""
        return command.command_type == VoiceCommandType.ANALYSIS
    
    async def handle(
        self,
        command: VoiceCommand,
        session: VoiceSession
    ) -> VoiceResponse:
        """Handle analytics command"""
        
        # Generate mock insights
        text = """Based on the last 7 days of data, I've identified a 15% increase in data collection efficiency.
        The peak performance hours are between 2 AM and 6 AM UTC.
        I recommend scheduling more intensive scrapers during these hours."""
        
        return VoiceResponse(
            text=text,
            emotion=EmotionalTone.PROFESSIONAL,
            visualizations=[
                {
                    'type': 'line_chart',
                    'data': 'efficiency_trend',
                    'title': '7-Day Efficiency Trend'
                }
            ],
            follow_up_questions=[
                "Would you like a detailed report?",
                "Should I adjust the scraper schedules?"
            ]
        )


class HelpSkill:
    """Skill for providing help"""
    
    name = "help"
    description = "Provide help and guidance"
    keywords = ["help", "how", "guide", "tutorial", "assist"]
    
    async def can_handle(self, command: VoiceCommand) -> bool:
        """Check if can handle command"""
        return command.command_type == VoiceCommandType.HELP
    
    async def handle(
        self,
        command: VoiceCommand,
        session: VoiceSession
    ) -> VoiceResponse:
        """Handle help command"""
        
        text = """I can help you with:
        - Checking system status and health
        - Managing scrapers (start, stop, schedule)
        - Analyzing data and generating reports
        - Configuring alerts and monitoring
        - And much more!
        
        Just ask me what you'd like to do."""
        
        return VoiceResponse(
            text=text,
            emotion=EmotionalTone.FRIENDLY,
            follow_up_questions=[
                "What would you like to know more about?",
                "Should I show you some examples?"
            ]
        )


class VoiceAIAssistant:
    """Main Voice AI Assistant for MCP Stack"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.nlu_engine = NLUEngine()
        self.speech_recognition = SpeechRecognitionEngine()
        self.text_to_speech = TextToSpeechEngine()
        self.conversation_manager = ConversationManager(database_url)
        self.response_generator = ResponseGenerator()
        self.skills_manager = SkillsManager()
        self.audio_buffer = []
        self.is_listening = False
    
    async def start_listening(
        self,
        user_id: str,
        continuous: bool = True,
        wake_word_enabled: bool = True
    ):
        """Start listening for voice commands"""
        
        # Create session
        session = await self.conversation_manager.create_session(user_id)
        
        self.is_listening = True
        logger.info(f"Voice assistant started for user {user_id}")
        
        if continuous:
            await self._continuous_listening(session, wake_word_enabled)
        else:
            await self._single_command(session)
    
    async def _continuous_listening(
        self,
        session: VoiceSession,
        wake_word_enabled: bool
    ):
        """Continuous listening mode"""
        
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024
        )
        
        logger.info("Listening... Say 'Hey MCP' to activate")
        
        try:
            while self.is_listening:
                # Read audio chunk
                audio_chunk = stream.read(1024, exception_on_overflow=False)
                
                # Check for speech
                if self.speech_recognition.is_speech(audio_chunk):
                    self.audio_buffer.append(audio_chunk)
                else:
                    if self.audio_buffer:
                        # Process accumulated audio
                        audio_data = b''.join(self.audio_buffer)
                        self.audio_buffer = []
                        
                        # Check for wake word or process command
                        if wake_word_enabled and not session.wake_word_detected:
                            if await self.speech_recognition.detect_wake_word(audio_data):
                                session.wake_word_detected = True
                                await self._respond("I'm listening", session)
                        else:
                            # Process command
                            await self._process_audio(audio_data, session)
                            
                            if wake_word_enabled:
                                session.wake_word_detected = False
                
                await asyncio.sleep(0.01)
                
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
    
    async def _single_command(self, session: VoiceSession):
        """Single command mode"""
        
        # Record audio
        audio_data = await self._record_audio(duration=5)
        
        # Process
        await self._process_audio(audio_data, session)
    
    async def _record_audio(self, duration: int = 5) -> bytes:
        """Record audio for specified duration"""
        
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=1024
        )
        
        frames = []
        for _ in range(0, int(16000 / 1024 * duration)):
            data = stream.read(1024)
            frames.append(data)
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        return b''.join(frames)
    
    async def _process_audio(self, audio_data: bytes, session: VoiceSession):
        """Process audio input"""
        
        with response_time.time():
            # Speech recognition
            text, confidence = await self.speech_recognition.recognize_speech(
                audio_data,
                session.language
            )
            
            if confidence < 0.3:
                await self._respond("I didn't quite catch that. Could you repeat?", session)
                return
            
            logger.info(f"Recognized: {text} (confidence: {confidence:.2f})")
            
            # Process text command
            await self.process_text_command(text, session)
    
    async def process_text_command(self, text: str, session: VoiceSession):
        """Process text command"""
        
        # NLU
        command = await self.nlu_engine.understand(text, session.language)
        
        # Update metrics
        voice_commands.labels(
            command.command_type.value,
            'processing'
        ).inc()
        
        # Log sentiment
        sentiment_scores.labels(
            'positive' if command.sentiment > 0 else 'negative'
        ).observe(abs(command.sentiment))
        
        # Check if confirmation needed
        if command.requires_confirmation:
            await self._request_confirmation(command, session)
            return
        
        # Find appropriate skill
        skill = await self.skills_manager.find_skill(command)
        
        if skill:
            # Execute skill
            response = await skill.handle(command, session)
        else:
            # Generate general response
            context = await self._gather_context(command, session)
            response = await self.response_generator.generate_response(
                command,
                context,
                session.emotional_tone
            )
        
        # Update conversation history
        await self.conversation_manager.update_context(
            session,
            text,
            response.text
        )
        
        # Send response
        await self._respond(response.text, session, response.emotion)
        
        # Execute actions if any
        if response.actions:
            await self._execute_actions(response.actions)
        
        # Show visualizations if any
        if response.visualizations:
            await self._show_visualizations(response.visualizations)
        
        # Update metrics
        voice_commands.labels(
            command.command_type.value,
            'success'
        ).inc()
    
    async def _request_confirmation(
        self,
        command: VoiceCommand,
        session: VoiceSession
    ):
        """Request confirmation for command"""
        
        confirmation_text = f"Just to confirm, you want me to {command.intent}. Is that correct?"
        await self._respond(confirmation_text, session, EmotionalTone.PROFESSIONAL)
        
        # Wait for response (simplified - in production would handle properly)
        # Store pending command in session context
        session.context['pending_command'] = command
    
    async def _gather_context(
        self,
        command: VoiceCommand,
        session: VoiceSession
    ) -> Dict[str, Any]:
        """Gather context for command"""
        
        context = {
            'user_id': session.user_id,
            'session_id': session.id,
            'command_type': command.command_type.value,
            'entities': command.entities
        }
        
        # Get relevant conversation history
        relevant_history = await self.conversation_manager.get_relevant_context(
            session,
            command.raw_text
        )
        
        if relevant_history:
            context['history'] = relevant_history
        
        # Add mock data based on command type
        if command.command_type == VoiceCommandType.QUERY:
            if 'scraper' in command.raw_text:
                context['result'] = "Found 3 active scrapers"
                context['count'] = 3
                context['details'] = "All scrapers are running normally"
        
        return context
    
    async def _respond(
        self,
        text: str,
        session: VoiceSession,
        emotion: EmotionalTone = EmotionalTone.NEUTRAL
    ):
        """Send response to user"""
        
        # Synthesize speech
        audio_data = await self.text_to_speech.synthesize_speech(
            text,
            session.language,
            emotion
        )
        
        # Play audio (simplified - in production would stream)
        self._play_audio(audio_data)
        
        logger.info(f"Assistant: {text}")
    
    def _play_audio(self, audio_data: bytes):
        """Play audio response"""
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_file.write(audio_data)
            tmp_path = tmp_file.name
        
        # Play using system command (simplified)
        if os.name == 'nt':  # Windows
            os.system(f'start {tmp_path}')
        else:  # Unix
            os.system(f'afplay {tmp_path} 2>/dev/null || aplay {tmp_path} 2>/dev/null')
        
        # Cleanup after delay
        asyncio.create_task(self._cleanup_audio_file(tmp_path))
    
    async def _cleanup_audio_file(self, path: str):
        """Clean up temporary audio file"""
        await asyncio.sleep(5)
        try:
            os.unlink(path)
        except:
            pass
    
    async def _execute_actions(self, actions: List[Dict[str, Any]]):
        """Execute actions from response"""
        
        for action in actions:
            logger.info(f"Executing action: {action}")
            
            # In production, would actually execute the actions
            # This is where integration with MCP Stack happens
            
            if action['type'] == 'manage_scraper':
                # Call scraper management API
                pass
            elif action['type'] == 'generate_report':
                # Call report generation API
                pass
    
    async def _show_visualizations(self, visualizations: List[Dict[str, Any]]):
        """Show visualizations"""
        
        for viz in visualizations:
            logger.info(f"Showing visualization: {viz}")
            
            # In production, would send to UI for display
            # Could use WebSocket to push to dashboard
    
    async def stop_listening(self):
        """Stop listening"""
        self.is_listening = False
        logger.info("Voice assistant stopped")


# Example usage
async def voice_demo():
    """Demo voice assistant functionality"""
    
    # Initialize assistant
    assistant = VoiceAIAssistant('postgresql://user:pass@localhost/voice_db')
    
    # Test text commands
    session = await assistant.conversation_manager.create_session("demo_user")
    
    # Test various commands
    test_commands = [
        "Hey MCP, what's the system status?",
        "Show me active scrapers",
        "Start the news scraper",
        "Analyze data trends for the last week",
        "Generate a performance report",
        "Help me configure alerts"
    ]
    
    for command in test_commands:
        print(f"\nUser: {command}")
        await assistant.process_text_command(command, session)
        await asyncio.sleep(2)
    
    # End session
    await assistant.conversation_manager.end_session(session.id)


if __name__ == "__main__":
    asyncio.run(voice_demo())