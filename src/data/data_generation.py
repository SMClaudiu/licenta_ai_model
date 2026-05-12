import psycopg2
import random
from faker import Faker
from datetime import datetime, timedelta
from src.core.data_config import scenarios, user_profiles, global_issues

DB_CONFIG = {
    "dbname": "licenta_db", "user": "postgres", "password": "101102",
    "host": "localhost", "port": "5432"
}
NUM_CLIENTS = 500  # Increased for more diversity
NUM_TASKS = 30000  # 10,000 per status
fake = Faker()

# Real-world companies and industries with current global issues



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
        print("\nData generation completed successfully!")
        print(f"Final status distribution: {status_counts}")
        print(f"Industries represented: {len(scenarios)}")
        print(f"User profiles: {len(user_profiles)}")
        print(f"Issues addressed: {len(global_issues)}")

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error: {error}")
        if conn is not None:
            conn.rollback()
    finally:
        if conn is not None:
            conn.close()
            print("Database connection closed.")


if __name__ == "__main__":
    generate_data()