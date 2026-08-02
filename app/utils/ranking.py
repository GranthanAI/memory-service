"""
app/utils/ranking.py

Thin wrapper/alias mapping to app/services/ranking_service.py.
Makes ranking functions available in both packages to support legacy imports.
"""

from app.services.ranking_service import RankingService

calculate_score = RankingService.calculate_score
rank_facts = RankingService.rank_facts
