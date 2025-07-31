import psycopg2
import random
import bcrypt
from faker import Faker
from datetime import datetime, timedelta
import numpy as np

DB_CONFIG = {
    "dbname": "licenta_db",
    "user": "postgres",
    "password": "101102",
    "host": "localhost",
    "port": "5432"
}

NUM_CLIENTS = 200
NUM_TASKS = 5000

fake = Faker()

# Enhanced corporate domains with realistic task patterns
CORPORATE_DOMAINS = {
    "it_dev": {
        "board_names": ["Backend Refactor Sprint", "API Gateway Migration", "Q3 Security Audit", "Production Hotfixes",
                        "UI/UX Overhaul"],
        "task_jargon": ["deploy to Staging", "fix bug #", "code review PR #", "optimize DB query for ",
                        "investigate latency in ", "refactor service "],
        "typical_duration_range": (1, 14),  # IT tasks are usually shorter
        "urgency_multiplier": 1.3  # IT tasks tend to be more urgent
    },
    "marketing": {
        "board_names": ["Q4 Social Media Campaign", "New Product Launch", "Content Calendar - Blog",
                        "SEO & SEM Strategy", "Brand Awareness Initiative"],
        "task_jargon": ["draft press release for ", "analyze campaign results of ", "create new ad copy for ",
                        "schedule social media posts for ", "A/B test landing page ", "update content brief for "],
        "typical_duration_range": (3, 21),  # Marketing tasks vary more
        "urgency_multiplier": 0.9
    },
    "sales": {
        "board_names": ["Q3 Lead Pipeline", "Key Account - Acme Corp", "Cold Outreach Strategy",
                        "Sales Team Weekly Sync", "New Market Expansion"],
        "task_jargon": ["follow up with client ", "prepare demo for prospect ", "update CRM with notes for ",
                        "log call with ", "research potential leads in ", "finalize contract for "],
        "typical_duration_range": (1, 7),  # Sales tasks are often quick
        "urgency_multiplier": 1.2
    },
    "hr": {
        "board_names": ["New Hire Onboarding", "Performance Review Cycle", "Company Culture Initiative",
                        "Recruitment - Senior Dev", "Benefits Package Review"],
        "task_jargon": ["schedule interview for ", "process payroll changes for ", "prepare benefits package for ",
                        "draft new company policy on ", "conduct exit interview with ", "organize team building event "],
        "typical_duration_range": (2, 14),  # HR tasks are moderate
        "urgency_multiplier": 0.8
    },
    "finance": {
        "board_names": ["End-of-Month Closing", "Budget Planning 2024", "Expense Report Approvals", "Annual Audit Prep",
                        "Investor Relations Deck"],
        "task_jargon": ["reconcile accounts for ", "process invoice #", "generate P&L report for ",
                        "approve expense report for ", "analyze financial variance for ", "forecast cash flow for "],
        "typical_duration_range": (1, 10),  # Finance tasks often have deadlines
        "urgency_multiplier": 1.1
    }
}

# Enhanced user profiles with realistic work behavior patterns
USER_PROFILES = {
    "beginner": {
        "weight": 35,
        "dashboards_per_client": (1, 1),
        "boards_per_dashboard": (1, 2),
        "task_name_prefix": ["Do", "Check", "Buy", "Call"],
        "description_length": (5, 50),
        "work_behavior": {
            "multitasking_tendency": 0.2,  # Low multitasking
            "procrastination_tendency": 0.4,  # Higher procrastination
            "weekend_work_probability": 0.05,
            "overtime_probability": 0.1,
            "task_completion_consistency": 0.6  # Less consistent
        }
    },
    "medium": {
        "weight": 50,
        "dashboards_per_client": (1, 3),
        "boards_per_dashboard": (2, 5),
        "task_name_prefix": ["Review", "Draft", "Implement", "Prepare", "Finalize"],
        "description_length": (50, 250),
        "work_behavior": {
            "multitasking_tendency": 0.6,  # Moderate multitasking
            "procrastination_tendency": 0.2,
            "weekend_work_probability": 0.15,
            "overtime_probability": 0.3,
            "task_completion_consistency": 0.8
        }
    },
    "advanced": {
        "weight": 15,
        "dashboards_per_client": (3, 7),
        "boards_per_dashboard": (5, 10),
        "task_name_prefix": ["URGENT: Fix", "BLOCKER:", "CRITICAL: Deploy", "Investigate", "Optimize"],
        "description_length": (250, 1000),
        "work_behavior": {
            "multitasking_tendency": 0.9,  # High multitasking
            "procrastination_tendency": 0.05,  # Very low procrastination
            "weekend_work_probability": 0.4,
            "overtime_probability": 0.6,
            "task_completion_consistency": 0.95  # Very consistent
        }
    }
}


def get_weighted_profile_name():
    """Get a user profile based on weights"""
    profiles = list(USER_PROFILES.keys())
    weights = [USER_PROFILES[p]['weight'] for p in profiles]
    return random.choices(profiles, weights, k=1)[0]


def calculate_realistic_task_status(task_data, client_behavior, current_time):
    """
    Calculate realistic task status based on multiple factors:
    - Task age and urgency
    - Client work behavior patterns
    - Task complexity and type
    - Time pressures
    """
    
    creation_date = task_data['creation_date']
    due_date = task_data['due_date']
    task_age_hours = (current_time - creation_date).total_seconds() / 3600
    task_age_days = task_age_hours / 24
    
    hours_until_due = (due_date - current_time).total_seconds() / 3600
    days_until_due = hours_until_due / 24
    
    # Calculate base probabilities
    is_overdue = hours_until_due < 0
    is_urgent = days_until_due <= 1
    is_very_urgent = hours_until_due <= 24
    
    # Task complexity factors
    has_priority_keywords = any(keyword in task_data['name'].lower() 
                               for keyword in ['urgent', 'critical', 'blocker', 'asap', 'emergency'])
    has_action_keywords = any(keyword in task_data['name'].lower()
                             for keyword in ['fix', 'deploy', 'implement', 'review', 'optimize'])
    
    # Client behavior factors
    multitask_tendency = client_behavior['multitasking_tendency']
    procrastination = client_behavior['procrastination_tendency']
    consistency = client_behavior['task_completion_consistency']
    
    # === STATUS CALCULATION LOGIC ===
    
    # COMPLETED (Status 2) - Realistic completion patterns
    completion_probability = 0.0
    
    if is_overdue:
        # Overdue tasks are more likely to be completed (catch-up work)
        completion_probability += 0.6 + (consistency * 0.3)
    elif task_age_days > 1:
        # Tasks older than 1 day have increasing completion probability
        completion_probability += min(0.4 + (task_age_days * 0.1), 0.8)
    
    if has_action_keywords:
        completion_probability += 0.2  # Action-oriented tasks get done faster
    
    if is_very_urgent and consistency > 0.7:
        completion_probability += 0.3  # Consistent workers complete urgent tasks
    
    # Apply consistency factor
    completion_probability *= consistency
    
    # IN PROGRESS (Status 1) - Key differentiator logic
    inprogress_probability = 0.0
    
    if task_age_hours >= 2 and not is_overdue:  # Task has been around for a bit
        inprogress_probability += 0.3
    
    if is_urgent and not is_overdue:
        inprogress_probability += 0.4  # Urgent tasks get started
    
    if has_priority_keywords:
        inprogress_probability += 0.3  # Priority tasks get attention
    
    if multitask_tendency > 0.5:
        inprogress_probability += 0.2  # Multitaskers have more in-progress tasks
    
    # Time-based in-progress patterns
    current_hour = current_time.hour
    if 9 <= current_hour <= 17:  # Business hours
        inprogress_probability += 0.2
    
    # Apply anti-procrastination factor
    inprogress_probability *= (1 - procrastination * 0.5)
    
    # PENDING (Status 0) - Default for new/unstarted tasks
    pending_probability = 1.0 - completion_probability - inprogress_probability
    
    # Ensure probabilities sum to 1 and are valid
    total_prob = completion_probability + inprogress_probability + pending_probability
    if total_prob > 0:
        completion_probability /= total_prob
        inprogress_probability /= total_prob
        pending_probability /= total_prob
    
    # Additional business rules
    if task_age_hours < 1:
        # Very new tasks are mostly pending
        return 0 if random.random() < 0.8 else (1 if random.random() < 0.7 else 2)
    
    if is_overdue and random.random() < 0.7:
        # Most overdue tasks should be completed or in urgent progress
        return 2 if random.random() < 0.6 else 1
    
    # Final status decision based on calculated probabilities
    rand = random.random(   )
    if rand < completion_probability:
        return 2  # Completed
    elif rand < completion_probability + inprogress_probability:
        return 1  # In Progress
    else:
        return 0  # Pending


def generate_realistic_data():
    """Generate realistic task management data with proper status patterns"""
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        print("Connected to database.")

        print("Clearing existing tables...")
        cur.execute("TRUNCATE TABLE task, board, dashboard, client RESTART IDENTITY CASCADE;")
        conn.commit()

        clients_data = []
        boards_data = []
        current_time = datetime.now()

        print(f"Generating {NUM_CLIENTS} clients with realistic profiles...")
        for _ in range(NUM_CLIENTS):
            profile_name = get_weighted_profile_name()
            domain_name = random.choice(list(CORPORATE_DOMAINS.keys()))

            password = fake.password()

            cur.execute(
                """
                INSERT INTO client (email, name, password, phone_number)
                VALUES (%s, %s, %s, %s) RETURNING client_id;
                """,
                (fake.email(), fake.name(), password, fake.phone_number())
            )
            client_id = cur.fetchone()[0]
            
            clients_data.append({
                "client_id": client_id, 
                "profile": profile_name, 
                "domain": domain_name,
                "behavior": USER_PROFILES[profile_name]["work_behavior"]
            })

        print("Generating dashboards and boards...")
        for client in clients_data:
            profile = USER_PROFILES[client["profile"]]
            domain_data = CORPORATE_DOMAINS[client["domain"]]

            num_dashboards = random.randint(*profile["dashboards_per_client"])
            for _ in range(num_dashboards):
                dashboard_name = f"{random.choice(domain_data['board_names'])} Dashboard"
                cur.execute(
                    "INSERT INTO dashboard (name, client_id) VALUES (%s, %s) RETURNING id;",
                    (dashboard_name, client["client_id"])
                )
                dashboard_id = cur.fetchone()[0]

                num_boards = random.randint(*profile["boards_per_dashboard"])
                for _ in range(num_boards):
                    board_name = random.choice(domain_data['board_names'])
                    cur.execute(
                        "INSERT INTO board (name, dash_board_id) VALUES (%s, %s) RETURNING board_id;",
                        (board_name, dashboard_id)
                    )
                    board_id = cur.fetchone()[0]
                    
                    boards_data.append({
                        "board_id": board_id, 
                        "profile": client["profile"], 
                        "domain": client["domain"],
                        "client_behavior": client["behavior"]
                    })

        print(f"Generating {NUM_TASKS} tasks with realistic status patterns...")
        
        # Track status distribution for verification
        status_counts = {0: 0, 1: 0, 2: 0}
        
        for i in range(NUM_TASKS):
            board_info = random.choice(boards_data)
            board_id = board_info["board_id"]
            profile_name = board_info["profile"]
            domain_name = board_info["domain"]
            client_behavior = board_info["client_behavior"]

            profile = USER_PROFILES[profile_name]
            domain_data = CORPORATE_DOMAINS[domain_name]

            # Generate task details
            task_prefix = random.choice(profile['task_name_prefix'])
            task_jargon = random.choice(domain_data['task_jargon'])

            specific_entity = ""
            if "#" in task_jargon:
                specific_entity = str(random.randint(100, 9999))
            else:
                specific_entity = fake.company() if random.random() > 0.5 else fake.name()

            task_name = f"{task_prefix} {task_jargon}{specific_entity}"
            description = fake.text(max_nb_chars=random.randint(*profile["description_length"]))

            # Generate realistic timing
            creation_date = current_time - timedelta(
                days=random.randint(1, 180),  # Tasks created in last 6 months
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )

            # Determine task duration based on domain and urgency
            duration_range = domain_data['typical_duration_range']
            urgency_mult = domain_data['urgency_multiplier']
            
            base_duration = random.randint(*duration_range)
            
            # Adjust duration based on task priority keywords
            if any(keyword in task_name.lower() for keyword in ['urgent', 'critical', 'blocker']):
                base_duration = max(1, int(base_duration * 0.5))  # Urgent tasks have shorter deadlines
            
            # Apply domain urgency multiplier
            final_duration = max(1, int(base_duration * urgency_mult))
            
            due_date = creation_date + timedelta(days=final_duration)

            # Calculate realistic status
            task_data = {
                'name': task_name,
                'creation_date': creation_date,
                'due_date': due_date
            }
            
            status = calculate_realistic_task_status(task_data, client_behavior, current_time)
            status_counts[status] += 1

            # Insert task
            cur.execute(
                """
                INSERT INTO task (creation_date, description, due_date, name, status, board_id)
                VALUES (%s, %s, %s, %s, %s, %s);
                """,
                (creation_date, description, due_date, task_name, status, board_id)
            )

            if (i + 1) % 500 == 0:
                print(f"  ... {i + 1}/{NUM_TASKS} tasks generated.")
                # Print current status distribution
                total_so_far = sum(status_counts.values())
                print(f"    Status distribution so far: "
                      f"Pending: {status_counts[0]/total_so_far*100:.1f}%, "
                      f"In Progress: {status_counts[1]/total_so_far*100:.1f}%, "
                      f"Completed: {status_counts[2]/total_so_far*100:.1f}%")

        conn.commit()
        cur.close()
        
        # Final status distribution
        total_tasks = sum(status_counts.values())
        print(f"\nFinal Status Distribution:")
        print(f"  Pending (0): {status_counts[0]} ({status_counts[0]/total_tasks*100:.1f}%)")
        print(f"  In Progress (1): {status_counts[1]} ({status_counts[1]/total_tasks*100:.1f}%)")
        print(f"  Completed (2): {status_counts[2]} ({status_counts[2]/total_tasks*100:.1f}%)")
        
        print("\nRealistic data generation completed successfully!")
        print("Key improvements:")
        print("- Status assignment based on task age, urgency, and client behavior")
        print("- Work behavior patterns (multitasking, procrastination, consistency)")
        print("- Domain-specific task durations and urgency patterns")
        print("- Time-aware status calculation (business hours, overdue logic)")
        print("- Realistic task naming with priority keywords")

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error: {error}")
        if conn is not None:
            conn.rollback()
    finally:
        if conn is not None:
            conn.close()
            print("Database connection closed.")


if __name__ == "__main__":
    generate_realistic_data()