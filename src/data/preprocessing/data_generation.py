import psycopg2
import random
from faker import Faker
from datetime import datetime, timedelta

DB_CONFIG = {
    "dbname": "licenta_db", "user": "postgres", "password": "101102",
    "host": "localhost", "port": "5432"
}
NUM_CLIENTS = 500  # Increased for more diversity
NUM_TASKS = 30000  # 10,000 per status
fake = Faker()

# Real-world companies and industries with current global issues
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


def get_weighted_profile_name():
    profiles = list(user_profiles.keys())
    weights = [user_profiles[p]['weight'] for p in profiles]
    return random.choices(profiles, weights, k=1)[0]


def generate_task_details(industry_key, profile_name):
    """Generate task name and description based on industry and profile"""
    industry = scenarios[industry_key]
    profile = user_profiles[profile_name]

    # Choose between industry-specific issue or global issue
    if random.random() < 0.7:
        issue = random.choice(industry["current_issues"])
    else:
        issue = random.choice(global_issues)

    # Generate task name using template
    template = random.choice(industry["task_templates"])
    task_name = template.format(issue)

    # Add urgency markers for senior roles
    if profile["work_behavior"]["urgent_task_probability"] > random.random():
        urgency_markers = ["URGENT:", "CRITICAL:", "HIGH PRIORITY:", "ESCALATED:", "IMMEDIATE ACTION:"]
        task_name = f"{random.choice(urgency_markers)} {task_name}"

    # Generate comprehensive description
    company = random.choice(industry["companies"])
    description_parts = [
        f"Company: {company}",
        f"Context: {issue.title()} initiative requiring immediate attention.",
        f"Objective: {fake.catch_phrase()}",
        f"Key deliverable: {fake.bs()}",
        f"Success criteria: {fake.bs()}",
    ]

    # Add stakeholders for senior roles
    if profile_name in ["senior_level", "executive_level"]:
        stakeholders = [fake.name() for _ in range(random.randint(2, 5))]
        description_parts.append(f"Key stakeholders: {', '.join(stakeholders)}")

    # Add budget considerations for executive level
    if profile_name == "executive_level":
        budget = f"${random.randint(50, 5000)}K"
        description_parts.append(f"Budget allocation: {budget}")

    description = " | ".join(description_parts)

    # Ensure description meets length requirements
    min_len, max_len = profile["description_length"]
    while len(description) < min_len:
        description += f" Additional notes: {fake.sentence()}"

    if len(description) > max_len:
        description = description[:max_len - 3] + "..."

    return task_name, description


def generate_dates_for_status(target_status, behavior, avg_duration_range, is_urgent=False):
    """Generate dates that correspond logically to a target status"""
    consistency = behavior['task_completion_consistency']
    procrastination = behavior['procrastination_tendency']
    current_time = datetime.now()

    min_duration, max_duration = avg_duration_range
    base_duration_days = random.uniform(min_duration, max_duration)

    # Urgent tasks have shorter durations
    if is_urgent:
        base_duration_days *= 0.3

    effective_duration_days = base_duration_days * (1 + (1 - consistency) * 0.5)

    if target_status == 0:  # Pending
        percentage_passed = random.uniform(0.01, 0.4) * (1 + procrastination)
        percentage_passed = min(percentage_passed, 0.5)
        days_since_creation = effective_duration_days * percentage_passed
        creation_date = current_time - timedelta(days=days_since_creation)
        due_date = creation_date + timedelta(days=effective_duration_days)

    elif target_status == 1:  # In Progress
        multitasking_factor = behavior['multitasking_tendency'] * 0.2
        percentage_passed = random.uniform(0.4 - multitasking_factor, 0.85 - multitasking_factor)
        days_since_creation = effective_duration_days * percentage_passed
        creation_date = current_time - timedelta(days=days_since_creation)
        due_date = creation_date + timedelta(days=effective_duration_days)

    else:  # Completed
        is_completed_late = random.random() > consistency
        percentage_passed = random.uniform(1.1, 1.4) if is_completed_late else random.uniform(0.85, 1.05)
        days_since_creation = effective_duration_days * percentage_passed
        creation_date = current_time - timedelta(days=days_since_creation)
        due_date = creation_date + timedelta(days=effective_duration_days)

    # Adjust for weekend work probability
    if behavior['weekend_work_probability'] < random.random():
        # Avoid weekend dates for creation
        while creation_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
            creation_date += timedelta(days=1)

    return creation_date, due_date


def generate_data():
    conn = None
    status_counts = {0: 0, 1: 0, 2: 0}

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        print("Connected to database.")

        print("Clearing existing tables...")
        cur.execute("TRUNCATE TABLE task, board, dashboard, client RESTART IDENTITY CASCADE;")
        conn.commit()

        clients_data = []
        boards_data = []

        print(f"Generating {NUM_CLIENTS} clients with profiles...")

        # Generate clients with industry assignments
        industry_keys = list(scenarios.keys())

        for _ in range(NUM_CLIENTS):
            profile_name = get_weighted_profile_name()
            industry_key = random.choice(industry_keys)

            # Generate professional email
            industry_data = scenarios[industry_key]
            company = random.choice(industry_data["companies"])
            domain = company.lower().replace(" ", "").replace("-", "") + ".com"

            first_name = fake.first_name()
            last_name = fake.last_name()
            email = f"{first_name.lower()}.{last_name.lower()}@{domain}"
            full_name = f"{first_name} {last_name}"

            cur.execute(
                "INSERT INTO client (email, name, password, phone_number) VALUES (%s, %s, %s, %s) RETURNING client_id;",
                (email, full_name, fake.password(), fake.phone_number())
            )
            client_id = cur.fetchone()[0]
            clients_data.append({
                "client_id": client_id,
                "profile": profile_name,
                "industry": industry_key,
                "company": company
            })

        print("Generating dashboards and boards...")
        for client in clients_data:
            profile = user_profiles[client["profile"]]
            industry_data = scenarios[client["industry"]]

            num_dashboards = random.randint(*profile["dashboards_per_client"])
            for _ in range(num_dashboards):
                dashboard_name = f"{client['company']} - {random.choice(industry_data['board_names'])}"
                cur.execute(
                    "INSERT INTO dashboard (name, client_id) VALUES (%s, %s) RETURNING id;",
                    (dashboard_name, client["client_id"])
                )
                dashboard_id = cur.fetchone()[0]

                num_boards = random.randint(*profile["boards_per_dashboard"])
                for _ in range(num_boards):
                    board_name = random.choice(industry_data['board_names'])
                    cur.execute(
                        "INSERT INTO board (name, dash_board_id) VALUES (%s, %s) RETURNING board_id;",
                        (board_name, dashboard_id)
                    )
                    board_id = cur.fetchone()[0]
                    boards_data.append({
                        "board_id": board_id,
                        "profile": client["profile"],
                        "industry": client["industry"],
                        "company": client["company"]
                    })

        print(f"Generating {NUM_TASKS} tasks...")

        for i in range(NUM_TASKS):
            # Ensure balanced distribution
            target_status = i % 3

            board_info = random.choice(boards_data)
            board_id = board_info["board_id"]
            profile_name = board_info["profile"]
            industry_key = board_info["industry"]

            profile = user_profiles[profile_name]
            industry_data = scenarios[industry_key]
            behavior = profile["work_behavior"]

            # Generate task details
            task_name, description = generate_task_details(industry_key, profile_name)

            # Check if task is urgent
            is_urgent = "URGENT:" in task_name or "CRITICAL:" in task_name

            # Generate dates
            creation_date, due_date = generate_dates_for_status(
                target_status,
                behavior,
                industry_data["average_duration_range"],
                is_urgent
            )

            status = target_status
            status_counts[status] += 1

            cur.execute(
                "INSERT INTO task (creation_date, description, due_date, name, status, board_id) VALUES (%s, %s, %s, %s, %s, %s);",
                (creation_date, description, due_date, task_name, status, board_id)
            )

            if (i + 1) % 1000 == 0:
                print(f"  ... {i + 1}/{NUM_TASKS} tasks generated.")

        conn.commit()
        cur.close()
        print("\n✅ Data generation completed successfully!")
        print(f"📊 Final status distribution: {status_counts}")
        print(f"🏢 Industries represented: {len(scenarios)}")
        print(f"👥 User profiles: {len(user_profiles)}")
        print(f"🌍 Issues addressed: {len(global_issues)}")

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"❌ Error: {error}")
        if conn is not None:
            conn.rollback()
    finally:
        if conn is not None:
            conn.close()
            print("Database connection closed.")


if __name__ == "__main__":
    generate_data()