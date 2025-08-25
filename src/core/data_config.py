scenarios = {
    "tech_giants": {
        "companies": ["Meta", "Google", "Apple", "Microsoft", "Amazon", "Netflix", "Tesla", "NVIDIA", "OpenAI",
                      "Anthropic"],
        "board_names": [
            "AI Ethics & Safety Board", "Climate Neutrality Initiative", "Data Privacy Compliance",
            "Metaverse Development", "Cloud Infrastructure Migration", "Cybersecurity Response Team",
            "Remote Work Optimization", "Supply Chain Resilience", "Digital Transformation"
        ],
        "current_issues": [
            "AI regulation compliance", "carbon footprint reduction", "data sovereignty laws",
            "supply chain disruptions", "chip shortage mitigation", "remote work policies",
            "cybersecurity threats", "talent retention crisis", "economic recession preparation"
        ],
        "task_templates": [
            "Implement {} for Q4 compliance deadline",
            "Address {} in European markets",
            "Develop strategy for {} impact on operations",
            "Crisis management: Handle {} situation",
            "Stakeholder meeting: Discuss {} implications",
            "Legal review: {} regulatory requirements",
            "Risk assessment: {} potential outcomes",
            "Budget allocation for {} initiatives"
        ],
        "average_duration_range": (2, 30)
    },

    "healthcare_pharma": {
        "companies": ["Pfizer", "Johnson & Johnson", "Moderna", "AstraZeneca", "Roche", "Novartis", "Merck", "GSK"],
        "board_names": [
            "Clinical Trials Management", "Regulatory Affairs", "Drug Development Pipeline",
            "Global Health Access", "Manufacturing Quality Control", "Post-Market Surveillance",
            "Digital Health Innovation", "Pandemic Preparedness"
        ],
        "current_issues": [
            "Long COVID treatment research", "vaccine distribution equity", "drug pricing transparency",
            "clinical trial diversity", "regulatory approval delays", "supply chain vulnerabilities",
            "mental health crisis response", "antimicrobial resistance", "healthcare worker shortage"
        ],
        "task_templates": [
            "Clinical study: Investigate {} treatment protocols",
            "Regulatory submission for {} approval",
            "Manufacturing scale-up for {} production",
            "Patient safety monitoring: {} adverse events",
            "Market access strategy for {} in developing countries",
            "Research collaboration on {} with academic institutions",
            "Quality assurance audit for {} processes",
            "Global health initiative: Address {} disparities"
        ],
        "average_duration_range": (7, 180)
    },

    "financial_services": {
        "companies": ["JPMorgan Chase", "Goldman Sachs", "Bank of America", "Wells Fargo", "Citigroup",
                      "Morgan Stanley", "BlackRock", "Visa"],
        "board_names": [
            "Digital Banking Transformation", "ESG Investment Strategy", "Regulatory Compliance",
            "Cybersecurity & Fraud Prevention", "Credit Risk Management", "Fintech Integration",
            "Climate Risk Assessment", "Customer Experience Innovation"
        ],
        "current_issues": [
            "inflation impact on portfolios", "cryptocurrency regulation", "ESG investment mandates",
            "banking sector consolidation", "interest rate volatility", "digital payment security",
            "climate change financial risks", "geopolitical investment risks", "regulatory stress testing"
        ],
        "task_templates": [
            "Risk model update: Incorporate {} factors",
            "Compliance audit: {} regulatory changes",
            "Client advisory: {} market conditions",
            "Investment strategy: Hedge against {} risks",
            "Due diligence: {} acquisition opportunity",
            "Stress testing: {} scenario analysis",
            "Product development: {} financial instruments",
            "ESG screening: Evaluate {} criteria"
        ],
        "average_duration_range": (1, 90)
    },

    "energy_utilities": {
        "companies": ["ExxonMobil", "Shell", "BP", "TotalEnergies", "Chevron", "NextEra Energy", "Engie", "Ørsted"],
        "board_names": [
            "Renewable Energy Transition", "Carbon Capture Projects", "Grid Modernization",
            "Energy Security Planning", "Environmental Compliance", "Clean Technology R&D",
            "Stakeholder Relations", "Emergency Response Team"
        ],
        "current_issues": [
            "net zero emissions targets", "energy price volatility", "grid stability challenges",
            "renewable energy integration", "carbon tax implications", "geopolitical supply risks",
            "extreme weather impacts", "energy storage solutions", "just transition policies"
        ],
        "task_templates": [
            "Feasibility study: {} implementation timeline",
            "Environmental impact assessment for {}",
            "Stakeholder consultation on {} project",
            "Technical analysis: {} integration challenges",
            "Regulatory filing for {} permit application",
            "Risk mitigation plan for {} scenarios",
            "Investment proposal: {} infrastructure upgrade",
            "Community engagement: {} local impact"
        ],
        "average_duration_range": (14, 365)
    },

    "retail_consumer": {
        "companies": ["Amazon", "Walmart", "Target", "Costco", "Home Depot", "Nike", "Unilever", "Procter & Gamble"],
        "board_names": [
            "Supply Chain Optimization", "Sustainable Packaging Initiative", "Digital Commerce Evolution",
            "Customer Experience Enhancement", "Inventory Management", "Brand Portfolio Strategy",
            "Global Market Expansion", "Social Impact Programs"
        ],
        "current_issues": [
            "supply chain bottlenecks", "consumer behavior shifts", "inflation cost pressures",
            "sustainability packaging demands", "labor shortage impacts", "e-commerce competition",
            "social responsibility expectations", "economic downturn preparation", "inventory optimization"
        ],
        "task_templates": [
            "Market research: {} consumer sentiment analysis",
            "Supply chain audit: {} vendor relationships",
            "Product development: {} sustainable alternatives",
            "Pricing strategy: Navigate {} cost increases",
            "Digital transformation: {} customer journey optimization",
            "Brand campaign: Address {} social issues",
            "Operational efficiency: {} process improvement",
            "Competitive analysis: {} market positioning"
        ],
        "average_duration_range": (3, 120)
    },

    "automotive_mobility": {
        "companies": ["Tesla", "Toyota", "Volkswagen", "GM", "Ford", "BMW", "Mercedes-Benz", "Uber", "Lyft"],
        "board_names": [
            "Electric Vehicle Development", "Autonomous Driving Technology", "Manufacturing Efficiency",
            "Battery Technology R&D", "Charging Infrastructure", "Mobility Services Innovation",
            "Supply Chain Security", "Regulatory Compliance"
        ],
        "current_issues": [
            "EV market competition", "battery supply constraints", "autonomous vehicle regulation",
            "charging infrastructure gaps", "semiconductor shortage", "raw material sourcing",
            "carbon emission standards", "consumer adoption barriers", "manufacturing capacity"
        ],
        "task_templates": [
            "R&D project: {} technology advancement",
            "Manufacturing optimization: {} production efficiency",
            "Regulatory compliance: {} safety standards",
            "Partnership negotiation: {} strategic alliance",
            "Market analysis: {} competitive landscape",
            "Supply chain diversification: {} risk mitigation",
            "Product testing: {} performance validation",
            "Consumer education: {} adoption strategy"
        ],
        "average_duration_range": (7, 540)
    },

    "telecommunications": {
        "companies": ["Verizon", "AT&T", "T-Mobile", "Vodafone", "China Mobile", "Deutsche Telekom", "Orange",
                      "Telefonica"],
        "board_names": [
            "5G Network Deployment", "Cybersecurity Infrastructure", "Digital Services Innovation",
            "Network Modernization", "Customer Experience", "Regulatory Affairs",
            "Spectrum Management", "IoT Solutions Development"
        ],
        "current_issues": [
            "5G rollout challenges", "network security threats", "spectrum auction costs",
            "infrastructure investment needs", "regulatory compliance burden", "competitive pricing pressure",
            "rural connectivity gaps", "technology obsolescence", "customer churn management"
        ],
        "task_templates": [
            "Network deployment: {} coverage expansion",
            "Security assessment: {} vulnerability testing",
            "Service launch: {} product development",
            "Infrastructure upgrade: {} technology migration",
            "Regulatory response: {} policy compliance",
            "Customer retention: {} loyalty program",
            "Technical support: {} issue resolution",
            "Strategic planning: {} market opportunity"
        ],
        "average_duration_range": (5, 180)
    },

    "aerospace_defense": {
        "companies": ["Boeing", "Airbus", "Lockheed Martin", "Raytheon", "Northrop Grumman", "SpaceX", "Blue Origin"],
        "board_names": [
            "Space Exploration Programs", "Defense Contract Management", "Aviation Safety Board",
            "Advanced Manufacturing", "Satellite Technology", "Mission Critical Systems",
            "International Partnerships", "Sustainability Initiatives"
        ],
        "current_issues": [
            "space debris management", "aviation safety concerns", "defense budget constraints",
            "supply chain security", "international sanctions impact", "climate change adaptation",
            "workforce skills gap", "technology export controls", "satellite constellation management"
        ],
        "task_templates": [
            "Mission planning: {} trajectory optimization",
            "Safety review: {} incident investigation",
            "Contract negotiation: {} defense proposal",
            "Technical specification: {} system requirements",
            "Quality assurance: {} manufacturing standards",
            "International coordination: {} partnership agreement",
            "Risk assessment: {} mission critical analysis",
            "Innovation project: {} next-generation technology"
        ],
        "average_duration_range": (14, 1095)
    }
}

# Enhanced user profiles with more behaviors
user_profiles= {
     "entry_level":
         {"weight": 25,
          "dashboards_per_client": (1, 2),
          "boards_per_dashboard": (1, 3),
          "description_length": (20, 150),
          "work_behavior":
              {"multitasking_tendency": 0.3,
               "procrastination_tendency": 0.5,
               "weekend_work_probability": 0.1,
               "overtime_probability": 0.2,
               "task_completion_consistency":0.6,
               "urgent_task_probability": 0.1
              }
          },
    "mid_level":{
        "weight":40 ,
        "dashboards_per_client":(2 , 4) ,
        "boards_per_dashboard":(3 , 6) ,
        "description_length":(100 , 400) ,
        "work_behavior":{
            "multitasking_tendency":0.7 ,
            "procrastination_tendency":0.25 ,
            "weekend_work_probability":0.25 ,
            "overtime_probability":0.4 ,
            "task_completion_consistency":0.75 ,
            "urgent_task_probability":0.2
        }
    } ,
    "senior_level":{
        "weight":25 ,
        "dashboards_per_client":(3 , 6) ,
        "boards_per_dashboard":(4 , 8) ,
        "description_length":(200 , 800) ,
        "work_behavior":{
            "multitasking_tendency":0.9 ,
            "procrastination_tendency":0.1 ,
            "weekend_work_probability":0.4 ,
            "overtime_probability":0.6 ,
            "task_completion_consistency":0.85 ,
            "urgent_task_probability":0.3
        }
    } ,
    "executive_level":{
        "weight":10 ,
        "dashboards_per_client":(4 , 8) ,
        "boards_per_dashboard":(6 , 12) ,
        "description_length":(300 , 1200) ,
        "work_behavior":{
            "multitasking_tendency":0.95 ,
            "procrastination_tendency":0.05 ,
            "weekend_work_probability":0.6 ,
            "overtime_probability":0.8 ,
            "task_completion_consistency":0.95 ,
            "urgent_task_probability":0.4
        }
    }
}

# Issues that affect all industries
global_issues = [
    "climate change adaptation" , "cybersecurity threats" , "supply chain disruptions" , "economic uncertainty" ,
    "talent acquisition" , "regulatory compliance" ,
    "digital transformation" , "sustainability reporting" , "geopolitical risks" , "inflation impacts" ,
    "remote work optimization" , "AI integration" ,
    "data privacy protection" , "energy transition" , "social responsibility"
]