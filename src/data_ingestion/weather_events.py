"""
Weather & Natural Event Monitoring
Tracks weather alerts and natural disasters
"""

import asyncio
from typing import List, Dict
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class WeatherAlert:
    """Data model for weather alerts"""
    id: str
    alert_type: str  # flood, cyclone, drought, heat_wave, etc.
    severity: str  # low, medium, high, extreme
    region: str
    description: str
    issued_at: datetime
    valid_until: datetime
    source: str
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['issued_at'] = self.issued_at.isoformat()
        data['valid_until'] = self.valid_until.isoformat()
        return data


class MetDepartmentMonitor:
    """
    Monitor Department of Meteorology Sri Lanka
    """
    
    def __init__(self):
        self.base_url = "http://www.meteo.gov.lk"
        
    async def get_active_warnings(self) -> List[WeatherAlert]:
        """Fetch active weather warnings"""
        # Simulated data (actual implementation would scrape website)
        alerts = []
        
        # Example alerts
        sample_alerts = [
            {
                "alert_type": "heavy_rain",
                "severity": "medium",
                "region": "Western Province",
                "description": "Heavy rainfall expected 50-75mm in next 24 hours",
                "hours_valid": 24
            },
            {
                "alert_type": "strong_winds",
                "severity": "low",
                "region": "Coastal Areas",
                "description": "Wind speed may increase up to 40-50 kmph",
                "hours_valid": 12
            }
        ]
        
        for i, alert_data in enumerate(sample_alerts):
            alert = WeatherAlert(
                id=f"alert_{datetime.now().strftime('%Y%m%d')}_{i}",
                alert_type=alert_data["alert_type"],
                severity=alert_data["severity"],
                region=alert_data["region"],
                description=alert_data["description"],
                issued_at=datetime.now(),
                valid_until=datetime.now() + timedelta(hours=alert_data["hours_valid"]),
                source="Department of Meteorology"
            )
            alerts.append(alert)
        
        logger.info(f"Retrieved {len(alerts)} weather alerts")
        return alerts
    
    async def get_weather_forecast(self) -> Dict:
        """Get general weather forecast"""
        return {
            "timestamp": datetime.now().isoformat(),
            "regions": {
                "Western": {"condition": "Partly Cloudy", "temp_c": 28, "rain_chance": 60},
                "Central": {"condition": "Cloudy", "temp_c": 22, "rain_chance": 80},
                "Southern": {"condition": "Sunny", "temp_c": 30, "rain_chance": 20},
                "Northern": {"condition": "Partly Cloudy", "temp_c": 32, "rain_chance": 30},
                "Eastern": {"condition": "Showers", "temp_c": 29, "rain_chance": 70}
            }
        }


class DisasterManagementMonitor:
    """
    Monitor Disaster Management Centre
    """
    
    async def get_active_disasters(self) -> List[Dict]:
        """Get information on active disasters"""
        # Simulated disaster tracking
        disasters = [
            {
                "type": "flood",
                "status": "ongoing",
                "affected_districts": ["Gampaha", "Colombo"],
                "families_affected": 250,
                "severity": "medium",
                "reported_at": (datetime.now() - timedelta(days=1)).isoformat()
            }
        ]
        
        logger.info(f"Retrieved {len(disasters)} active disaster records")
        return disasters
    
    async def get_evacuation_centers(self) -> List[Dict]:
        """Get active evacuation centers"""
        centers = [
            {
                "name": "Community Center - Kaduwela",
                "district": "Colombo",
                "capacity": 500,
                "current_occupancy": 120,
                "status": "active"
            }
        ]
        
        return centers


class SeismicMonitor:
    """
    Monitor seismic activity
    """
    
    async def get_recent_earthquakes(self, hours: int = 24) -> List[Dict]:
        """Get earthquakes in Sri Lanka region"""
        # Simulated data (actual would use USGS or similar)
        earthquakes = [
            {
                "magnitude": 3.2,
                "location": "75km SW of Colombo",
                "depth_km": 10,
                "time": (datetime.now() - timedelta(hours=6)).isoformat(),
                "felt": False
            }
        ]
        
        return earthquakes


class WeatherEventAggregator:
    """Aggregates all weather and natural event data"""
    
    def __init__(self):
        self.met_dept = MetDepartmentMonitor()
        self.disaster_mgmt = DisasterManagementMonitor()
        self.seismic = SeismicMonitor()
        
    async def collect_all(self) -> Dict:
        """Collect all weather and disaster data"""
        logger.info("Collecting weather and disaster data...")
        
        data = {
            "collection_time": datetime.now().isoformat(),
            "weather_alerts": [alert.to_dict() for alert in await self.met_dept.get_active_warnings()],
            "weather_forecast": await self.met_dept.get_weather_forecast(),
            "active_disasters": await self.disaster_mgmt.get_active_disasters(),
            "evacuation_centers": await self.disaster_mgmt.get_evacuation_centers(),
            "recent_earthquakes": await self.seismic.get_recent_earthquakes()
        }
        
        logger.info("Weather and disaster data collection completed")
        return data
    
    def calculate_weather_risk_score(self, data: Dict) -> float:
        """
        Calculate overall weather/disaster risk
        0 = no risk, 1 = extreme risk
        """
        risk_score = 0.0
        
        # Weather alerts
        alerts = data.get("weather_alerts", [])
        if alerts:
            severity_weights = {"low": 0.2, "medium": 0.5, "high": 0.8, "extreme": 1.0}
            alert_scores = [severity_weights.get(alert["severity"], 0.3) for alert in alerts]
            risk_score += sum(alert_scores) / len(alerts) * 0.4
        
        # Active disasters
        disasters = data.get("active_disasters", [])
        if disasters:
            risk_score += min(len(disasters) * 0.2, 0.4)
        
        # Earthquakes
        earthquakes = data.get("recent_earthquakes", [])
        if earthquakes:
            for eq in earthquakes:
                if eq.get("magnitude", 0) > 4.0:
                    risk_score += 0.2
        
        return min(risk_score, 1.0)
    
    def generate_weather_summary(self, data: Dict) -> str:
        """Generate human-readable summary"""
        risk = self.calculate_weather_risk_score(data)
        
        if risk > 0.7:
            status = "HIGH RISK"
            emoji = "🔴"
        elif risk > 0.4:
            status = "MODERATE RISK"
            emoji = "🟡"
        else:
            status = "LOW RISK"
            emoji = "🟢"
        
        summary_lines = [
            f"{emoji} Weather Risk Status: {status} (Score: {risk:.2f})",
            ""
        ]
        
        # Active alerts
        alerts = data.get("weather_alerts", [])
        if alerts:
            summary_lines.append("⚠️ Active Weather Alerts:")
            for alert in alerts:
                summary_lines.append(f"  • [{alert['severity'].upper()}] {alert['region']}: {alert['description']}")
        else:
            summary_lines.append("✅ No active weather alerts")
        
        # Disasters
        disasters = data.get("active_disasters", [])
        if disasters:
            summary_lines.append("\n🚨 Active Disasters:")
            for disaster in disasters:
                summary_lines.append(f"  • {disaster['type'].title()} in {', '.join(disaster['affected_districts'])}")
                summary_lines.append(f"    {disaster['families_affected']} families affected")
        
        # Forecast summary
        forecast = data.get("weather_forecast", {})
        if "regions" in forecast:
            summary_lines.append("\n🌤️ Regional Forecast:")
            for region, info in forecast["regions"].items():
                summary_lines.append(f"  • {region}: {info['condition']}, {info['temp_c']}°C, {info['rain_chance']}% rain")
        
        return "\n".join(summary_lines)


async def main():
    """Test weather monitoring"""
    from pathlib import Path
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from config import RAW_DATA_DIR
    
    aggregator = WeatherEventAggregator()
    
    # Collect all data
    data = await aggregator.collect_all()
    
    # Calculate risk
    risk_score = aggregator.calculate_weather_risk_score(data)
    
    # Generate summary
    summary = aggregator.generate_weather_summary(data)
    
    # Save to file
    output_file = RAW_DATA_DIR / f"weather_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("WEATHER & DISASTER MONITORING COMPLETE")
    print("="*60)
    print(summary)
    print(f"\nData saved to: {output_file}")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())