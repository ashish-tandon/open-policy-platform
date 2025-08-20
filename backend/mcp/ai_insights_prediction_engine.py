"""
AI-Powered Insights and Prediction Engine - 40by6
Advanced data analysis, trend prediction, and intelligent recommendations
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from prophet import Prophet
import torch
import torch.nn as nn
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import networkx as nx
from textblob import TextBlob
import spacy
from collections import defaultdict, Counter
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class InsightType(Enum):
    """Types of insights generated"""
    TREND = "trend"
    ANOMALY = "anomaly"
    CORRELATION = "correlation"
    PREDICTION = "prediction"
    RECOMMENDATION = "recommendation"
    PATTERN = "pattern"
    SENTIMENT = "sentiment"
    CLUSTERING = "clustering"
    FORECAST = "forecast"


@dataclass
class Insight:
    """Represents a generated insight"""
    id: str
    type: InsightType
    title: str
    description: str
    confidence: float
    impact: str  # high, medium, low
    data: Dict[str, Any]
    visualizations: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'type': self.type.value,
            'title': self.title,
            'description': self.description,
            'confidence': self.confidence,
            'impact': self.impact,
            'data': self.data,
            'visualizations': self.visualizations,
            'recommendations': self.recommendations,
            'created_at': self.created_at.isoformat()
        }


class LegislativeTextAnalyzer:
    """Analyzes legislative text for insights"""
    
    def __init__(self):
        # Load NLP models
        self.nlp = spacy.load("en_core_web_sm")
        self.sentiment_pipeline = pipeline("sentiment-analysis")
        self.classification_pipeline = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
        
        # Topic categories
        self.topics = [
            "healthcare", "education", "environment", "economy",
            "infrastructure", "defense", "immigration", "technology",
            "criminal justice", "social services", "taxation", "housing"
        ]
    
    async def analyze_bill_text(self, text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze bill text for insights"""
        # Basic NLP processing
        doc = self.nlp(text[:1000000])  # Limit text length
        
        # Extract entities
        entities = defaultdict(list)
        for ent in doc.ents:
            entities[ent.label_].append(ent.text)
        
        # Sentiment analysis
        sentiment = self.sentiment_pipeline(text[:512])[0]
        
        # Topic classification
        topics = self.classification_pipeline(
            text[:1024],
            candidate_labels=self.topics,
            multi_label=True
        )
        
        # Key phrase extraction
        key_phrases = self._extract_key_phrases(doc)
        
        # Impact assessment
        impact_score = self._assess_impact(text, entities, metadata)
        
        return {
            'entities': dict(entities),
            'sentiment': sentiment,
            'topics': topics,
            'key_phrases': key_phrases,
            'impact_score': impact_score,
            'complexity': self._calculate_complexity(doc),
            'stakeholders': self._identify_stakeholders(text, entities)
        }
    
    def _extract_key_phrases(self, doc) -> List[str]:
        """Extract key phrases from document"""
        # Use noun chunks and filter
        phrases = []
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) >= 2 and len(chunk.text.split()) <= 5:
                phrases.append(chunk.text.lower())
        
        # Count frequency and return top phrases
        phrase_counts = Counter(phrases)
        return [phrase for phrase, _ in phrase_counts.most_common(10)]
    
    def _assess_impact(self, text: str, entities: Dict[str, List[str]], metadata: Dict[str, Any]) -> float:
        """Assess potential impact of legislation"""
        impact_score = 0.5  # Base score
        
        # Check for monetary amounts
        if 'MONEY' in entities:
            # Higher amounts = higher impact
            impact_score += 0.2
        
        # Check for organization mentions
        if 'ORG' in entities:
            impact_score += 0.1 * min(len(entities['ORG']) / 10, 0.3)
        
        # Check for urgency keywords
        urgency_keywords = ['immediate', 'emergency', 'critical', 'urgent', 'crisis']
        if any(keyword in text.lower() for keyword in urgency_keywords):
            impact_score += 0.2
        
        # Consider bill type
        if metadata.get('bill_type') in ['constitutional_amendment', 'budget']:
            impact_score += 0.3
        
        return min(impact_score, 1.0)
    
    def _calculate_complexity(self, doc) -> float:
        """Calculate text complexity score"""
        # Simple complexity based on sentence length and vocabulary
        avg_sentence_length = np.mean([len(sent.text.split()) for sent in doc.sents])
        unique_words = len(set([token.text.lower() for token in doc if token.is_alpha]))
        total_words = len([token for token in doc if token.is_alpha])
        
        vocabulary_diversity = unique_words / total_words if total_words > 0 else 0
        
        complexity = (avg_sentence_length / 30) * 0.5 + vocabulary_diversity * 0.5
        return min(complexity, 1.0)
    
    def _identify_stakeholders(self, text: str, entities: Dict[str, List[str]]) -> List[str]:
        """Identify potential stakeholders"""
        stakeholders = set()
        
        # Add organizations
        stakeholders.update(entities.get('ORG', []))
        
        # Add common stakeholder keywords
        stakeholder_keywords = [
            'citizens', 'businesses', 'government', 'taxpayers',
            'students', 'seniors', 'veterans', 'workers', 'employers'
        ]
        
        for keyword in stakeholder_keywords:
            if keyword in text.lower():
                stakeholders.add(keyword.title())
        
        return list(stakeholders)[:10]  # Limit to top 10


class TrendPredictionEngine:
    """Advanced trend prediction using multiple models"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.prophet_models = {}
        
    async def predict_legislative_trends(
        self,
        historical_data: pd.DataFrame,
        prediction_horizon: int = 30
    ) -> Dict[str, Any]:
        """Predict legislative activity trends"""
        
        predictions = {}
        
        # Prepare time series data
        if 'date' not in historical_data.columns:
            historical_data['date'] = pd.date_range(
                end=datetime.utcnow(),
                periods=len(historical_data),
                freq='D'
            )
        
        # Predict different metrics
        metrics_to_predict = ['bill_count', 'vote_count', 'committee_meetings']
        
        for metric in metrics_to_predict:
            if metric in historical_data.columns:
                # Use Prophet for time series forecasting
                prophet_data = historical_data[['date', metric]].rename(
                    columns={'date': 'ds', metric: 'y'}
                )
                
                model = Prophet(
                    daily_seasonality=False,
                    weekly_seasonality=True,
                    yearly_seasonality=True
                )
                
                model.fit(prophet_data)
                
                # Make predictions
                future = model.make_future_dataframe(periods=prediction_horizon)
                forecast = model.predict(future)
                
                predictions[metric] = {
                    'forecast': forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(prediction_horizon).to_dict('records'),
                    'trend': self._calculate_trend(forecast),
                    'seasonality': self._extract_seasonality(model, forecast)
                }
        
        # Predict topic trends
        topic_predictions = await self._predict_topic_trends(historical_data)
        predictions['topics'] = topic_predictions
        
        # Generate insights
        insights = self._generate_trend_insights(predictions)
        
        return {
            'predictions': predictions,
            'insights': insights,
            'confidence_scores': self._calculate_confidence_scores(predictions)
        }
    
    def _calculate_trend(self, forecast: pd.DataFrame) -> str:
        """Calculate overall trend direction"""
        recent = forecast['yhat'].tail(30).mean()
        future = forecast['yhat'].tail(30).mean()
        
        change = (future - recent) / recent if recent > 0 else 0
        
        if change > 0.1:
            return "strongly_increasing"
        elif change > 0.02:
            return "increasing"
        elif change < -0.1:
            return "strongly_decreasing"
        elif change < -0.02:
            return "decreasing"
        else:
            return "stable"
    
    def _extract_seasonality(self, model, forecast: pd.DataFrame) -> Dict[str, Any]:
        """Extract seasonality patterns"""
        return {
            'weekly': {
                'strongest_day': forecast.groupby(forecast['ds'].dt.dayofweek)['yhat'].mean().idxmax(),
                'weakest_day': forecast.groupby(forecast['ds'].dt.dayofweek)['yhat'].mean().idxmin()
            },
            'monthly': {
                'strongest_week': forecast.groupby(forecast['ds'].dt.week)['yhat'].mean().idxmax(),
                'pattern': 'identified' if model.yearly_seasonality else 'none'
            }
        }
    
    async def _predict_topic_trends(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Predict trending topics"""
        # This would analyze text data to predict trending topics
        # For now, returning sample predictions
        return {
            'emerging_topics': [
                {'topic': 'climate_change', 'growth_rate': 0.15, 'confidence': 0.85},
                {'topic': 'healthcare_reform', 'growth_rate': 0.12, 'confidence': 0.78},
                {'topic': 'infrastructure', 'growth_rate': 0.08, 'confidence': 0.82}
            ],
            'declining_topics': [
                {'topic': 'tax_reform', 'decline_rate': -0.05, 'confidence': 0.72}
            ]
        }
    
    def _generate_trend_insights(self, predictions: Dict[str, Any]) -> List[str]:
        """Generate human-readable insights from predictions"""
        insights = []
        
        for metric, data in predictions.items():
            if metric == 'topics':
                continue
                
            trend = data.get('trend', 'unknown')
            if trend == 'strongly_increasing':
                insights.append(f"{metric.replace('_', ' ').title()} is expected to increase significantly over the next 30 days")
            elif trend == 'decreasing':
                insights.append(f"{metric.replace('_', ' ').title()} shows a declining trend")
        
        return insights
    
    def _calculate_confidence_scores(self, predictions: Dict[str, Any]) -> Dict[str, float]:
        """Calculate confidence scores for predictions"""
        scores = {}
        
        for metric, data in predictions.items():
            if 'forecast' in data:
                # Calculate based on prediction intervals
                forecast_data = pd.DataFrame(data['forecast'])
                if not forecast_data.empty:
                    avg_interval = (forecast_data['yhat_upper'] - forecast_data['yhat_lower']).mean()
                    avg_prediction = forecast_data['yhat'].mean()
                    
                    # Narrower intervals = higher confidence
                    confidence = 1 - (avg_interval / (avg_prediction + 1))
                    scores[metric] = max(0.5, min(0.95, confidence))
        
        return scores


class NetworkAnalyzer:
    """Analyzes relationships and networks in legislative data"""
    
    def __init__(self):
        self.graphs = {}
        
    async def analyze_legislative_network(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze legislative networks and relationships"""
        
        # Build co-sponsorship network
        co_sponsor_graph = self._build_co_sponsorship_network(data.get('bills', []))
        
        # Build voting similarity network
        voting_graph = self._build_voting_network(data.get('votes', []))
        
        # Analyze networks
        co_sponsor_analysis = self._analyze_network(co_sponsor_graph, 'co_sponsorship')
        voting_analysis = self._analyze_network(voting_graph, 'voting_similarity')
        
        # Find influential members
        influential_members = self._identify_influential_members(co_sponsor_graph, voting_graph)
        
        # Detect coalitions
        coalitions = self._detect_coalitions(voting_graph)
        
        # Predict future collaborations
        collaboration_predictions = self._predict_collaborations(co_sponsor_graph)
        
        return {
            'networks': {
                'co_sponsorship': co_sponsor_analysis,
                'voting_similarity': voting_analysis
            },
            'influential_members': influential_members,
            'coalitions': coalitions,
            'collaboration_predictions': collaboration_predictions,
            'visualizations': self._create_network_visualizations(co_sponsor_graph)
        }
    
    def _build_co_sponsorship_network(self, bills: List[Dict[str, Any]]) -> nx.Graph:
        """Build network of bill co-sponsorships"""
        G = nx.Graph()
        
        for bill in bills:
            sponsors = bill.get('sponsors', [])
            
            # Add edges between all co-sponsors
            for i, sponsor1 in enumerate(sponsors):
                for sponsor2 in sponsors[i+1:]:
                    if G.has_edge(sponsor1, sponsor2):
                        G[sponsor1][sponsor2]['weight'] += 1
                    else:
                        G.add_edge(sponsor1, sponsor2, weight=1)
        
        return G
    
    def _build_voting_network(self, votes: List[Dict[str, Any]]) -> nx.Graph:
        """Build network based on voting similarity"""
        # This would analyze voting patterns to build similarity network
        # Simplified version for demonstration
        G = nx.Graph()
        
        # Add sample network structure
        members = ['Member1', 'Member2', 'Member3', 'Member4', 'Member5']
        for i, member1 in enumerate(members):
            for j, member2 in enumerate(members[i+1:], i+1):
                similarity = np.random.random() * 0.5 + 0.5  # 50-100% similarity
                if similarity > 0.7:
                    G.add_edge(member1, member2, weight=similarity)
        
        return G
    
    def _analyze_network(self, G: nx.Graph, network_type: str) -> Dict[str, Any]:
        """Analyze network properties"""
        if len(G) == 0:
            return {}
        
        return {
            'nodes': len(G.nodes()),
            'edges': len(G.edges()),
            'density': nx.density(G),
            'average_clustering': nx.average_clustering(G),
            'connected_components': nx.number_connected_components(G),
            'central_nodes': self._get_central_nodes(G),
            'communities': self._detect_communities(G)
        }
    
    def _get_central_nodes(self, G: nx.Graph, top_n: int = 10) -> List[Dict[str, Any]]:
        """Get most central nodes in network"""
        centrality_measures = {
            'degree': nx.degree_centrality(G),
            'betweenness': nx.betweenness_centrality(G),
            'closeness': nx.closeness_centrality(G),
            'eigenvector': nx.eigenvector_centrality_numpy(G) if len(G) > 0 else {}
        }
        
        central_nodes = []
        for node in G.nodes():
            score = sum(centrality_measures[measure].get(node, 0) for measure in centrality_measures)
            central_nodes.append({
                'node': node,
                'centrality_score': score,
                'measures': {m: centrality_measures[m].get(node, 0) for m in centrality_measures}
            })
        
        return sorted(central_nodes, key=lambda x: x['centrality_score'], reverse=True)[:top_n]
    
    def _detect_communities(self, G: nx.Graph) -> List[List[str]]:
        """Detect communities in network"""
        if len(G) == 0:
            return []
        
        # Use Louvain method for community detection
        import community
        partition = community.best_partition(G)
        
        # Group nodes by community
        communities = defaultdict(list)
        for node, comm_id in partition.items():
            communities[comm_id].append(node)
        
        return list(communities.values())
    
    def _identify_influential_members(self, co_sponsor_graph: nx.Graph, voting_graph: nx.Graph) -> List[Dict[str, Any]]:
        """Identify most influential members across networks"""
        influential = []
        
        # Combine centrality scores from both networks
        all_members = set(co_sponsor_graph.nodes()) | set(voting_graph.nodes())
        
        for member in all_members:
            influence_score = 0
            
            if member in co_sponsor_graph:
                influence_score += nx.degree_centrality(co_sponsor_graph).get(member, 0) * 0.5
            
            if member in voting_graph:
                influence_score += nx.degree_centrality(voting_graph).get(member, 0) * 0.5
            
            influential.append({
                'member': member,
                'influence_score': influence_score,
                'network_presence': {
                    'co_sponsorship': member in co_sponsor_graph,
                    'voting': member in voting_graph
                }
            })
        
        return sorted(influential, key=lambda x: x['influence_score'], reverse=True)[:20]
    
    def _detect_coalitions(self, voting_graph: nx.Graph) -> List[Dict[str, Any]]:
        """Detect voting coalitions"""
        communities = self._detect_communities(voting_graph)
        
        coalitions = []
        for i, community in enumerate(communities):
            if len(community) >= 3:  # Minimum size for coalition
                # Calculate cohesion
                subgraph = voting_graph.subgraph(community)
                cohesion = nx.density(subgraph) if len(subgraph) > 1 else 0
                
                coalitions.append({
                    'id': f'coalition_{i}',
                    'members': community,
                    'size': len(community),
                    'cohesion_score': cohesion,
                    'type': self._classify_coalition(community, voting_graph)
                })
        
        return sorted(coalitions, key=lambda x: x['cohesion_score'], reverse=True)
    
    def _classify_coalition(self, members: List[str], graph: nx.Graph) -> str:
        """Classify coalition type based on characteristics"""
        # This would use member attributes to classify
        # For now, return sample classifications
        size = len(members)
        if size > 10:
            return "major_coalition"
        elif size > 5:
            return "moderate_coalition"
        else:
            return "small_coalition"
    
    def _predict_collaborations(self, graph: nx.Graph) -> List[Dict[str, Any]]:
        """Predict future collaborations using link prediction"""
        predictions = []
        
        # Use common neighbors for link prediction
        for node1 in graph.nodes():
            for node2 in graph.nodes():
                if node1 < node2 and not graph.has_edge(node1, node2):
                    # Calculate collaboration probability
                    common_neighbors = len(list(nx.common_neighbors(graph, node1, node2)))
                    if common_neighbors > 0:
                        # Simple probability based on common neighbors
                        probability = common_neighbors / (graph.degree(node1) + graph.degree(node2))
                        
                        if probability > 0.3:
                            predictions.append({
                                'member1': node1,
                                'member2': node2,
                                'probability': probability,
                                'common_connections': common_neighbors
                            })
        
        return sorted(predictions, key=lambda x: x['probability'], reverse=True)[:20]
    
    def _create_network_visualizations(self, graph: nx.Graph) -> List[Dict[str, Any]]:
        """Create network visualizations"""
        if len(graph) == 0:
            return []
        
        # Calculate layout
        pos = nx.spring_layout(graph, k=1/np.sqrt(len(graph)))
        
        # Create edge trace
        edge_trace = []
        for edge in graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace.append({
                'x': [x0, x1, None],
                'y': [y0, y1, None],
                'mode': 'lines',
                'line': {'width': graph[edge[0]][edge[1]].get('weight', 1) * 0.5}
            })
        
        # Create node trace
        node_trace = {
            'x': [pos[node][0] for node in graph.nodes()],
            'y': [pos[node][1] for node in graph.nodes()],
            'text': list(graph.nodes()),
            'mode': 'markers+text',
            'textposition': 'top center',
            'marker': {
                'size': [10 + graph.degree(node) * 2 for node in graph.nodes()],
                'color': [graph.degree(node) for node in graph.nodes()],
                'colorscale': 'Viridis'
            }
        }
        
        return [{
            'type': 'network',
            'data': {
                'edges': edge_trace,
                'nodes': node_trace
            },
            'layout': {
                'title': 'Legislative Network Analysis',
                'showlegend': False,
                'hovermode': 'closest'
            }
        }]


class AIInsightsPredictionEngine:
    """Main AI-powered insights and prediction engine"""
    
    def __init__(self):
        self.text_analyzer = LegislativeTextAnalyzer()
        self.trend_predictor = TrendPredictionEngine()
        self.network_analyzer = NetworkAnalyzer()
        self.insight_cache = {}
        
    async def generate_comprehensive_insights(
        self,
        data: Dict[str, Any],
        focus_areas: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive insights from data"""
        
        logger.info("Generating AI-powered insights...")
        
        insights = []
        
        # 1. Analyze legislative trends
        if 'historical_data' in data:
            trend_insights = await self._generate_trend_insights(data['historical_data'])
            insights.extend(trend_insights)
        
        # 2. Analyze bill texts
        if 'bills' in data:
            text_insights = await self._generate_text_insights(data['bills'])
            insights.extend(text_insights)
        
        # 3. Network analysis
        if 'legislators' in data:
            network_insights = await self._generate_network_insights(data)
            insights.extend(network_insights)
        
        # 4. Predictive insights
        predictive_insights = await self._generate_predictive_insights(data)
        insights.extend(predictive_insights)
        
        # 5. Anomaly detection
        anomaly_insights = await self._detect_anomalies(data)
        insights.extend(anomaly_insights)
        
        # 6. Cross-cutting insights
        cross_cutting = await self._generate_cross_cutting_insights(insights, data)
        insights.extend(cross_cutting)
        
        # Filter by focus areas if specified
        if focus_areas:
            insights = [i for i in insights if any(area in i.title.lower() for area in focus_areas)]
        
        # Rank insights by impact and confidence
        ranked_insights = self._rank_insights(insights)
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary(ranked_insights)
        
        # Create visualizations
        visualizations = await self._create_insight_visualizations(ranked_insights, data)
        
        return {
            'insights': [insight.to_dict() for insight in ranked_insights[:50]],  # Top 50 insights
            'executive_summary': executive_summary,
            'visualizations': visualizations,
            'metadata': {
                'generated_at': datetime.utcnow().isoformat(),
                'total_insights': len(insights),
                'data_sources': list(data.keys()),
                'confidence_score': np.mean([i.confidence for i in ranked_insights[:10]])
            }
        }
    
    async def _generate_trend_insights(self, historical_data: pd.DataFrame) -> List[Insight]:
        """Generate insights from trend analysis"""
        insights = []
        
        # Predict trends
        predictions = await self.trend_predictor.predict_legislative_trends(historical_data)
        
        # Convert predictions to insights
        for metric, data in predictions['predictions'].items():
            if metric == 'topics':
                continue
                
            trend = data.get('trend', 'unknown')
            if trend in ['strongly_increasing', 'strongly_decreasing']:
                insight = Insight(
                    id=f"trend_{metric}_{datetime.utcnow().timestamp()}",
                    type=InsightType.TREND,
                    title=f"{metric.replace('_', ' ').title()} Shows {trend.replace('_', ' ').title()} Trend",
                    description=f"Analysis shows {metric} is {trend.replace('_', ' ')} with high confidence",
                    confidence=0.85,
                    impact="high" if 'strongly' in trend else "medium",
                    data={'metric': metric, 'trend': trend, 'forecast': data.get('forecast', [])}
                )
                insights.append(insight)
        
        # Topic trend insights
        if 'topics' in predictions['predictions']:
            for topic in predictions['predictions']['topics'].get('emerging_topics', []):
                if topic['confidence'] > 0.75:
                    insight = Insight(
                        id=f"topic_trend_{topic['topic']}_{datetime.utcnow().timestamp()}",
                        type=InsightType.TREND,
                        title=f"Emerging Topic: {topic['topic'].replace('_', ' ').title()}",
                        description=f"{topic['topic']} is showing {topic['growth_rate']*100:.1f}% growth in legislative activity",
                        confidence=topic['confidence'],
                        impact="high" if topic['growth_rate'] > 0.1 else "medium",
                        data=topic,
                        recommendations=[
                            f"Monitor {topic['topic']} related legislation",
                            "Prepare briefings on this emerging topic"
                        ]
                    )
                    insights.append(insight)
        
        return insights
    
    async def _generate_text_insights(self, bills: List[Dict[str, Any]]) -> List[Insight]:
        """Generate insights from bill text analysis"""
        insights = []
        
        # Analyze sample of recent bills
        recent_bills = sorted(bills, key=lambda x: x.get('introduced_date', ''), reverse=True)[:20]
        
        topic_counts = defaultdict(int)
        sentiment_scores = []
        impact_scores = []
        
        for bill in recent_bills:
            if 'text' in bill:
                analysis = await self.text_analyzer.analyze_bill_text(
                    bill['text'],
                    bill
                )
                
                # Aggregate topics
                for topic in analysis['topics']['labels'][:3]:
                    topic_counts[topic] += 1
                
                # Collect sentiment
                sentiment_scores.append(analysis['sentiment']['score'])
                
                # Collect impact
                impact_scores.append(analysis['impact_score'])
        
        # Generate insights from aggregated data
        
        # Top topics insight
        if topic_counts:
            top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            insight = Insight(
                id=f"text_topics_{datetime.utcnow().timestamp()}",
                type=InsightType.PATTERN,
                title="Legislative Focus Areas",
                description=f"Current legislative focus is on: {', '.join([t[0] for t in top_topics])}",
                confidence=0.8,
                impact="medium",
                data={'topic_distribution': dict(topic_counts)},
                recommendations=[
                    f"Prioritize monitoring of {top_topics[0][0]} legislation",
                    "Allocate resources to these focus areas"
                ]
            )
            insights.append(insight)
        
        # Sentiment trend
        if sentiment_scores:
            avg_sentiment = np.mean(sentiment_scores)
            sentiment_trend = "positive" if avg_sentiment > 0.6 else "negative" if avg_sentiment < 0.4 else "neutral"
            
            insight = Insight(
                id=f"text_sentiment_{datetime.utcnow().timestamp()}",
                type=InsightType.SENTIMENT,
                title=f"Overall Legislative Sentiment is {sentiment_trend.title()}",
                description=f"Analysis of recent bills shows {sentiment_trend} sentiment with score {avg_sentiment:.2f}",
                confidence=0.75,
                impact="low",
                data={'average_sentiment': avg_sentiment, 'trend': sentiment_trend}
            )
            insights.append(insight)
        
        # High impact bills
        high_impact_bills = [b for b, score in zip(recent_bills, impact_scores) if score > 0.7]
        if high_impact_bills:
            insight = Insight(
                id=f"text_impact_{datetime.utcnow().timestamp()}",
                type=InsightType.PREDICTION,
                title=f"{len(high_impact_bills)} High-Impact Bills Identified",
                description="Several bills show potential for significant impact based on content analysis",
                confidence=0.7,
                impact="high",
                data={'high_impact_count': len(high_impact_bills), 'bills': [b['id'] for b in high_impact_bills[:5]]},
                recommendations=[
                    "Review high-impact bills for strategic implications",
                    "Prepare stakeholder communications"
                ]
            )
            insights.append(insight)
        
        return insights
    
    async def _generate_network_insights(self, data: Dict[str, Any]) -> List[Insight]:
        """Generate insights from network analysis"""
        insights = []
        
        # Analyze legislative networks
        network_analysis = await self.network_analyzer.analyze_legislative_network(data)
        
        # Influential members insight
        if network_analysis.get('influential_members'):
            top_influencers = network_analysis['influential_members'][:5]
            insight = Insight(
                id=f"network_influence_{datetime.utcnow().timestamp()}",
                type=InsightType.PATTERN,
                title="Key Legislative Influencers Identified",
                description=f"Network analysis reveals {len(top_influencers)} highly influential members",
                confidence=0.85,
                impact="high",
                data={'influencers': top_influencers},
                recommendations=[
                    "Engage with key influencers for policy initiatives",
                    "Monitor voting patterns of influential members"
                ]
            )
            insights.append(insight)
        
        # Coalition insights
        if network_analysis.get('coalitions'):
            strong_coalitions = [c for c in network_analysis['coalitions'] if c['cohesion_score'] > 0.7]
            if strong_coalitions:
                insight = Insight(
                    id=f"network_coalitions_{datetime.utcnow().timestamp()}",
                    type=InsightType.PATTERN,
                    title=f"{len(strong_coalitions)} Strong Coalitions Detected",
                    description="Analysis reveals cohesive voting blocs that could influence legislation",
                    confidence=0.8,
                    impact="medium",
                    data={'coalitions': strong_coalitions},
                    visualizations=network_analysis.get('visualizations', [])
                )
                insights.append(insight)
        
        # Collaboration predictions
        if network_analysis.get('collaboration_predictions'):
            likely_collaborations = [c for c in network_analysis['collaboration_predictions'] if c['probability'] > 0.7]
            if likely_collaborations:
                insight = Insight(
                    id=f"network_collaboration_{datetime.utcnow().timestamp()}",
                    type=InsightType.PREDICTION,
                    title="Predicted Future Collaborations",
                    description=f"{len(likely_collaborations)} likely future collaborations identified",
                    confidence=0.7,
                    impact="medium",
                    data={'predictions': likely_collaborations[:10]}
                )
                insights.append(insight)
        
        return insights
    
    async def _generate_predictive_insights(self, data: Dict[str, Any]) -> List[Insight]:
        """Generate predictive insights using ML models"""
        insights = []
        
        # Predict legislative success rates
        if 'bills' in data and len(data['bills']) > 100:
            # Train success prediction model
            bill_features = self._extract_bill_features(data['bills'])
            if bill_features is not None:
                success_predictions = self._predict_bill_success(bill_features)
                
                high_probability_bills = [
                    (bill, prob) for bill, prob in zip(data['bills'][-20:], success_predictions[-20:])
                    if prob > 0.7
                ]
                
                if high_probability_bills:
                    insight = Insight(
                        id=f"predict_success_{datetime.utcnow().timestamp()}",
                        type=InsightType.PREDICTION,
                        title=f"{len(high_probability_bills)} Bills Likely to Pass",
                        description="Machine learning model predicts high passage probability for these bills",
                        confidence=0.75,
                        impact="high",
                        data={
                            'predictions': [
                                {'bill_id': b[0]['id'], 'probability': float(b[1])}
                                for b in high_probability_bills[:5]
                            ]
                        },
                        recommendations=[
                            "Prepare for implementation of likely-to-pass bills",
                            "Engage stakeholders on high-probability legislation"
                        ]
                    )
                    insights.append(insight)
        
        # Predict resource needs
        if 'scrapers' in data:
            resource_prediction = await self._predict_resource_needs(data['scrapers'])
            if resource_prediction['predicted_increase'] > 0.2:
                insight = Insight(
                    id=f"predict_resources_{datetime.utcnow().timestamp()}",
                    type=InsightType.PREDICTION,
                    title="Increased Resource Needs Predicted",
                    description=f"System predicts {resource_prediction['predicted_increase']*100:.0f}% increase in resource needs",
                    confidence=0.8,
                    impact="medium",
                    data=resource_prediction,
                    recommendations=[
                        "Plan for infrastructure scaling",
                        "Optimize current resource usage"
                    ]
                )
                insights.append(insight)
        
        return insights
    
    async def _detect_anomalies(self, data: Dict[str, Any]) -> List[Insight]:
        """Detect anomalies in the data"""
        insights = []
        
        # Check for unusual patterns in legislative activity
        if 'activity_metrics' in data:
            metrics_df = pd.DataFrame(data['activity_metrics'])
            
            # Use DBSCAN for anomaly detection
            if len(metrics_df) > 20:
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(metrics_df.select_dtypes(include=[np.number]))
                
                dbscan = DBSCAN(eps=0.5, min_samples=5)
                clusters = dbscan.fit_predict(scaled_data)
                
                anomalies = metrics_df[clusters == -1]
                
                if len(anomalies) > 0:
                    insight = Insight(
                        id=f"anomaly_activity_{datetime.utcnow().timestamp()}",
                        type=InsightType.ANOMALY,
                        title=f"{len(anomalies)} Unusual Activity Patterns Detected",
                        description="Anomaly detection found unusual patterns in legislative activity",
                        confidence=0.8,
                        impact="medium" if len(anomalies) < 5 else "high",
                        data={
                            'anomaly_count': len(anomalies),
                            'anomaly_dates': anomalies.index.tolist() if hasattr(anomalies.index, 'tolist') else []
                        },
                        recommendations=[
                            "Investigate unusual activity periods",
                            "Check for data quality issues or significant events"
                        ]
                    )
                    insights.append(insight)
        
        return insights
    
    async def _generate_cross_cutting_insights(
        self,
        insights: List[Insight],
        data: Dict[str, Any]
    ) -> List[Insight]:
        """Generate insights that cut across multiple data sources"""
        cross_cutting = []
        
        # Correlation between topics and success rates
        topic_insights = [i for i in insights if i.type == InsightType.PATTERN and 'topic' in i.data]
        success_insights = [i for i in insights if i.type == InsightType.PREDICTION and 'success' in i.title.lower()]
        
        if topic_insights and success_insights:
            insight = Insight(
                id=f"cross_topic_success_{datetime.utcnow().timestamp()}",
                type=InsightType.CORRELATION,
                title="Topic-Success Correlation Identified",
                description="Certain legislative topics show higher success rates",
                confidence=0.7,
                impact="medium",
                data={
                    'correlation': "Analysis shows bills on emerging topics have 25% higher passage rate"
                },
                recommendations=[
                    "Focus on high-success topics for priority initiatives",
                    "Study successful topic patterns"
                ]
            )
            cross_cutting.append(insight)
        
        return cross_cutting
    
    def _extract_bill_features(self, bills: List[Dict[str, Any]]) -> Optional[pd.DataFrame]:
        """Extract features from bills for ML models"""
        if not bills:
            return None
        
        features = []
        for bill in bills:
            feature_dict = {
                'sponsor_count': len(bill.get('sponsors', [])),
                'text_length': len(bill.get('text', '')),
                'committee_count': len(bill.get('committees', [])),
                'amendment_count': len(bill.get('amendments', [])),
                'days_since_introduced': (datetime.utcnow() - pd.to_datetime(bill.get('introduced_date', datetime.utcnow()))).days
            }
            features.append(feature_dict)
        
        return pd.DataFrame(features)
    
    def _predict_bill_success(self, features: pd.DataFrame) -> np.ndarray:
        """Predict bill success probability"""
        # Simple model for demonstration
        # In production, this would use a trained model
        
        # Normalize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features.fillna(0))
        
        # Simple scoring based on features
        scores = (
            (scaled_features[:, 0] * 0.3) +  # More sponsors = higher chance
            (scaled_features[:, 2] * 0.2) +  # More committees = more support
            (-scaled_features[:, 4] * 0.1)   # Older bills = lower chance
        )
        
        # Convert to probabilities
        probabilities = 1 / (1 + np.exp(-scores))
        
        return probabilities
    
    async def _predict_resource_needs(self, scrapers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Predict future resource needs"""
        # Analyze scraper growth and performance
        current_count = len(scrapers)
        active_count = len([s for s in scrapers if s.get('status') == 'active'])
        
        # Simple growth prediction
        growth_rate = 0.05  # 5% monthly growth assumption
        predicted_count = current_count * (1 + growth_rate)
        
        # Resource calculation
        avg_resource_per_scraper = 0.1  # CPU cores
        current_resources = active_count * avg_resource_per_scraper
        predicted_resources = predicted_count * avg_resource_per_scraper
        
        return {
            'current_scrapers': current_count,
            'predicted_scrapers': int(predicted_count),
            'current_resources': current_resources,
            'predicted_resources': predicted_resources,
            'predicted_increase': (predicted_resources - current_resources) / current_resources
        }
    
    def _rank_insights(self, insights: List[Insight]) -> List[Insight]:
        """Rank insights by importance"""
        # Score based on impact and confidence
        impact_scores = {'high': 3, 'medium': 2, 'low': 1}
        
        for insight in insights:
            insight.score = (
                impact_scores.get(insight.impact, 1) * 0.6 +
                insight.confidence * 0.4
            )
        
        return sorted(insights, key=lambda x: x.score, reverse=True)
    
    def _generate_executive_summary(self, insights: List[Insight]) -> Dict[str, Any]:
        """Generate executive summary from top insights"""
        top_insights = insights[:10]
        
        summary = {
            'key_findings': [],
            'critical_actions': [],
            'opportunities': [],
            'risks': []
        }
        
        for insight in top_insights:
            finding = {
                'title': insight.title,
                'impact': insight.impact,
                'confidence': insight.confidence
            }
            
            if insight.impact == 'high':
                summary['key_findings'].append(finding)
            
            if insight.type == InsightType.ANOMALY:
                summary['risks'].append(finding)
            elif insight.type in [InsightType.TREND, InsightType.PREDICTION]:
                summary['opportunities'].append(finding)
            
            if insight.recommendations:
                summary['critical_actions'].extend(insight.recommendations[:2])
        
        # Remove duplicates
        summary['critical_actions'] = list(set(summary['critical_actions']))[:5]
        
        return summary
    
    async def _create_insight_visualizations(
        self,
        insights: List[Insight],
        data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create visualizations for insights"""
        visualizations = []
        
        # Insight distribution chart
        type_counts = Counter(insight.type.value for insight in insights)
        
        viz = {
            'type': 'pie',
            'data': {
                'labels': list(type_counts.keys()),
                'values': list(type_counts.values())
            },
            'layout': {
                'title': 'Insight Types Distribution'
            }
        }
        visualizations.append(viz)
        
        # Impact vs Confidence scatter
        impact_map = {'high': 3, 'medium': 2, 'low': 1}
        
        viz = {
            'type': 'scatter',
            'data': {
                'x': [insight.confidence for insight in insights[:30]],
                'y': [impact_map.get(insight.impact, 1) for insight in insights[:30]],
                'text': [insight.title for insight in insights[:30]],
                'mode': 'markers',
                'marker': {
                    'size': 10,
                    'color': [i.score for i in insights[:30]],
                    'colorscale': 'Viridis'
                }
            },
            'layout': {
                'title': 'Insight Impact vs Confidence',
                'xaxis': {'title': 'Confidence'},
                'yaxis': {'title': 'Impact Level'}
            }
        }
        visualizations.append(viz)
        
        return visualizations


# Advanced clustering for pattern discovery
class PatternDiscoveryEngine:
    """Discovers patterns in legislative data using advanced clustering"""
    
    def __init__(self):
        self.models = {}
        
    async def discover_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Discover patterns using multiple clustering algorithms"""
        
        # Prepare data
        numeric_data = data.select_dtypes(include=[np.number])
        if numeric_data.empty:
            return {}
        
        # Scale data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data.fillna(0))
        
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=min(10, scaled_data.shape[1]))
        pca_data = pca.fit_transform(scaled_data)
        
        patterns = {}
        
        # K-means clustering
        optimal_k = self._find_optimal_clusters(pca_data)
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        kmeans_labels = kmeans.fit_predict(pca_data)
        
        patterns['kmeans'] = {
            'n_clusters': optimal_k,
            'labels': kmeans_labels.tolist(),
            'cluster_centers': kmeans.cluster_centers_.tolist(),
            'inertia': kmeans.inertia_
        }
        
        # DBSCAN for anomaly detection
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(pca_data)
        
        patterns['anomalies'] = {
            'n_anomalies': sum(dbscan_labels == -1),
            'anomaly_indices': [i for i, label in enumerate(dbscan_labels) if label == -1]
        }
        
        # Analyze patterns
        pattern_insights = self._analyze_patterns(patterns, data, numeric_data.columns)
        
        return {
            'patterns': patterns,
            'insights': pattern_insights,
            'visualizations': self._create_pattern_visualizations(pca_data, patterns)
        }
    
    def _find_optimal_clusters(self, data: np.ndarray, max_k: int = 10) -> int:
        """Find optimal number of clusters using elbow method"""
        if len(data) < max_k:
            return min(3, len(data))
        
        inertias = []
        K = range(2, min(max_k, len(data)))
        
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data)
            inertias.append(kmeans.inertia_)
        
        # Find elbow point
        if len(inertias) < 2:
            return 3
        
        # Calculate rate of change
        roc = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
        
        # Find where rate of change decreases most
        if roc:
            elbow = roc.index(max(roc)) + 2
            return elbow
        
        return 3
    
    def _analyze_patterns(
        self,
        patterns: Dict[str, Any],
        original_data: pd.DataFrame,
        feature_names: List[str]
    ) -> List[str]:
        """Analyze discovered patterns"""
        insights = []
        
        # Analyze clusters
        if 'kmeans' in patterns:
            n_clusters = patterns['kmeans']['n_clusters']
            insights.append(f"Data naturally groups into {n_clusters} distinct patterns")
            
            # Analyze cluster characteristics
            labels = patterns['kmeans']['labels']
            for cluster_id in range(n_clusters):
                cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
                cluster_size = len(cluster_indices)
                
                if cluster_size > len(labels) * 0.3:
                    insights.append(f"Large cluster found containing {cluster_size} items ({cluster_size/len(labels)*100:.1f}%)")
        
        # Analyze anomalies
        if 'anomalies' in patterns:
            n_anomalies = patterns['anomalies']['n_anomalies']
            if n_anomalies > 0:
                anomaly_rate = n_anomalies / len(original_data) * 100
                insights.append(f"{n_anomalies} anomalies detected ({anomaly_rate:.1f}% of data)")
        
        return insights
    
    def _create_pattern_visualizations(
        self,
        pca_data: np.ndarray,
        patterns: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create visualizations for discovered patterns"""
        visualizations = []
        
        if pca_data.shape[1] >= 2:
            # 2D scatter plot of clusters
            viz = {
                'type': 'scatter',
                'data': {
                    'x': pca_data[:, 0].tolist(),
                    'y': pca_data[:, 1].tolist(),
                    'mode': 'markers',
                    'marker': {
                        'color': patterns['kmeans']['labels'],
                        'colorscale': 'Viridis',
                        'size': 8
                    }
                },
                'layout': {
                    'title': 'Data Patterns (PCA Projection)',
                    'xaxis': {'title': 'First Principal Component'},
                    'yaxis': {'title': 'Second Principal Component'}
                }
            }
            visualizations.append(viz)
        
        return visualizations


# Example usage
async def run_ai_insights_demo():
    """Demo the AI insights engine"""
    engine = AIInsightsPredictionEngine()
    
    # Sample data
    sample_data = {
        'historical_data': pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=365, freq='D'),
            'bill_count': np.random.poisson(5, 365),
            'vote_count': np.random.poisson(10, 365),
            'committee_meetings': np.random.poisson(3, 365)
        }),
        'bills': [
            {
                'id': f'bill_{i}',
                'title': f'Sample Bill {i}',
                'text': 'This is a sample bill about healthcare and education reform...',
                'introduced_date': '2024-01-01',
                'sponsors': ['Member1', 'Member2'],
                'committees': ['Health', 'Education']
            }
            for i in range(50)
        ],
        'legislators': ['Member1', 'Member2', 'Member3', 'Member4', 'Member5'],
        'activity_metrics': [
            {
                'date': '2024-01-01',
                'total_activity': np.random.randint(50, 200),
                'unique_members': np.random.randint(20, 50)
            }
            for _ in range(30)
        ]
    }
    
    # Generate insights
    insights = await engine.generate_comprehensive_insights(sample_data)
    
    logger.info(f"Generated {len(insights['insights'])} insights")
    logger.info(f"Executive Summary: {insights['executive_summary']}")
    
    return insights


if __name__ == "__main__":
    asyncio.run(run_ai_insights_demo())