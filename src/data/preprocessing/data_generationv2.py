# data_generator.py (Varianta Finală, Îmbunătățită)

import psycopg2
import random
from faker import Faker
from datetime import datetime, timedelta

DB_CONFIG = {
    "dbname": "licenta_db", "user": "postgres", "password": "101102",
    "host": "localhost", "port": "5432"
}
NUM_CLIENTS = 200
NUM_TASKS = 6000  # 33% per status
fake = Faker()

CORPORATE_DOMAINS = {
    "it_dev": {
        "board_names": ["Backend Refactor Sprint", "API Gateway Migration", "Q3 Security Audit", "Production Hotfixes",
                        "UI/UX Overhaul"],
        "task_jargon": ["deploy to Staging", "fix bug #", "code review PR #", "optimize DB query for ",
                        "investigate latency in ", "refactor service "], "average_duration_range": (1, 14)},
    "marketing": {"board_names": ["Q4 Social Media Campaign", "New Product Launch", "Content Calendar - Blog",
                                  "SEO & SEM Strategy", "Brand Awareness Initiative"],
                  "task_jargon": ["draft press release for ", "analyze campaign results of ", "create new ad copy for ",
                                  "schedule social media posts for ", "A/B test landing page ",
                                  "update content brief for "], "average_duration_range": (3, 21)},
    "sales": {"board_names": ["Q3 Lead Pipeline", "Key Account - Acme Corp", "Cold Outreach Strategy",
                              "Sales Team Weekly Sync", "New Market Expansion"],
              "task_jargon": ["follow up with client ", "prepare demo for prospect ", "update CRM with notes for ",
                              "log call with ", "research potential leads in ", "finalize contract for "],
              "average_duration_range": (1, 7)},
    "hr": {"board_names": ["New Hire Onboarding", "Performance Review Cycle", "Company Culture Initiative",
                           "Recruitment - Senior Dev", "Benefits Package Review"],
           "task_jargon": ["schedule interview for ", "process payroll changes for ", "prepare benefits package for ",
                           "draft new company policy on ", "conduct exit interview with ",
                           "organize team building event "], "average_duration_range": (2, 14)},
    "finance": {
        "board_names": ["End-of-Month Closing", "Budget Planning 2024", "Expense Report Approvals", "Annual Audit Prep",
                        "Investor Relations Deck"],
        "task_jargon": ["reconcile accounts for ", "process invoice #", "generate P&L report for ",
                        "approve expense report for ", "analyze financial variance for ", "forecast cash flow for "],
        "average_duration_range": (1, 10)}
}

USER_PROFILES = {
    "beginner": {"weight": 35, "dashboards_per_client": (1, 1), "boards_per_dashboard": (1, 2),
                 "task_name_prefix": ["Do", "Check", "Buy", "Call"], "description_length": (5, 50),
                 "work_behavior": {"multitasking_tendency": 0.2, "procrastination_tendency": 0.4,
                                   "weekend_work_probability": 0.05, "overtime_probability": 0.1,
                                   "task_completion_consistency": 0.6}},
    "medium": {"weight": 50, "dashboards_per_client": (1, 3), "boards_per_dashboard": (2, 5),
               "task_name_prefix": ["Review", "Draft", "Implement", "Prepare", "Finalize"],
               "description_length": (50, 250),
               "work_behavior": {"multitasking_tendency": 0.6, "procrastination_tendency": 0.2,
                                 "weekend_work_probability": 0.15, "overtime_probability": 0.3,
                                 "task_completion_consistency": 0.8}},
    "advanced": {"weight": 15, "dashboards_per_client": (3, 7), "boards_per_dashboard": (5, 10),
                 "task_name_prefix": ["URGENT: Fix", "BLOCKER:", "CRITICAL: Deploy", "Investigate", "Optimize"],
                 "description_length": (250, 1000),
                 "work_behavior": {"multitasking_tendency": 0.9, "procrastination_tendency": 0.05,
                                   "weekend_work_probability": 0.4, "overtime_probability": 0.6,
                                   "task_completion_consistency": 0.95}}
}


def get_weighted_profile_name():
    profiles = list(USER_PROFILES.keys())
    weights = [USER_PROFILES[p]['weight'] for p in profiles]
    return random.choices(profiles, weights, k=1)[0]


def generate_realistic_dates_for_status(target_status, behavior, avg_duration_range):

    consistency = behavior['task_completion_consistency']
    procrastination = behavior['procrastination_tendency']
    multitasking = behavior['multitasking_tendency']
    current_time = datetime.now()

    min_duration, max_duration = avg_duration_range
    base_duration_days = random.uniform(min_duration, max_duration)

    # Durata mai ridicata in relatie cu consecventa scazuta
    effective_duration_days = base_duration_days * (1 + (1 - consistency) * 0.5)

    if target_status == 0:  # Pending
        # Procrastinarea micsoreaza sansa ca un task abia primit sa fie inceput
        percentage_passed = random.uniform(0.01, 0.3) * (1 + procrastination)
        percentage_passed = min(percentage_passed, 0.4)  # Plafonăm pentru a crea o graniță clară

        days_since_creation = effective_duration_days * percentage_passed
        creation_date = current_time - timedelta(days=days_since_creation)
        due_date = creation_date + timedelta(days=effective_duration_days)

    elif target_status == 1:  # In Progress
        # Multitaskingul mareste durata celor "In Progress"
        multitasking_factor = multitasking * 0.2
        percentage_passed = random.uniform(0.4, 0.8)  # Graniță clară față de "Pending"

        days_since_creation = effective_duration_days * percentage_passed
        creation_date = current_time - timedelta(days=days_since_creation)
        due_date = creation_date + timedelta(days=effective_duration_days)

    else:  # Completed
        # Pentru "Completed", task-ul este aproape gata sau terminat cu întârziere.
        is_completed_late = random.random() > consistency
        percentage_passed = random.uniform(1.05, 1.3) if is_completed_late else random.uniform(0.8, 1.0)

        days_since_creation = effective_duration_days * percentage_passed
        creation_date = current_time - timedelta(days=days_since_creation)
        due_date = creation_date + timedelta(days=effective_duration_days)

    return creation_date, due_date

def generate_data():
    conn = None
    status_counts = {0: 0, 1: 0, 2: 0}
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        print("Conectat la baza de date.")

        print("Curățarea tabelelor existente...")
        cur.execute("TRUNCATE TABLE task, board, dashboard, client RESTART IDENTITY CASCADE;")
        conn.commit()

        clients_data = []
        boards_data = []

        print(f"Generare {NUM_CLIENTS} clienti...")
        for _ in range(NUM_CLIENTS):
            profile_name = get_weighted_profile_name()
            domain_name = random.choice(list(CORPORATE_DOMAINS.keys()))
            password = fake.password()
            cur.execute(
                "INSERT INTO client (email, name, password, phone_number) VALUES (%s, %s, %s, %s) RETURNING client_id;",
                (fake.email(), fake.name(), password, fake.phone_number()))
            client_id = cur.fetchone()[0]
            clients_data.append({"client_id": client_id, "profile": profile_name, "domain": domain_name})

        print("Generare dashboards si boards...")
        for client in clients_data:
            profile = USER_PROFILES[client["profile"]]
            domain_data = CORPORATE_DOMAINS[client["domain"]]
            num_dashboards = random.randint(*profile["dashboards_per_client"])
            for _ in range(num_dashboards):
                dashboard_name = f"{random.choice(domain_data['board_names'])} Dashboard"
                cur.execute("INSERT INTO dashboard (name, client_id) VALUES (%s, %s) RETURNING id;",
                            (dashboard_name, client["client_id"]))
                dashboard_id = cur.fetchone()[0]
                num_boards = random.randint(*profile["boards_per_dashboard"])
                for _ in range(num_boards):
                    board_name = random.choice(domain_data['board_names'])
                    cur.execute("INSERT INTO board (name, dash_board_id) VALUES (%s, %s) RETURNING board_id;",
                                (board_name, dashboard_id))
                    board_id = cur.fetchone()[0]
                    boards_data.append({"board_id": board_id, "profile": client["profile"], "domain": client["domain"]})

        print(f"Generare {NUM_TASKS} task-uri echilibrate...")
        for i in range(NUM_TASKS):
            # MODIFICARE CHEIE: Decidem statusul ÎNAINTE de a genera datele pentru a asigura echilibrul
            target_status = i % 3

            board_info = random.choice(boards_data)
            board_id = board_info["board_id"]
            profile_name = board_info["profile"]
            domain_name = board_info["domain"]
            profile = USER_PROFILES[profile_name]
            domain_data = CORPORATE_DOMAINS[domain_name]
            behavior = profile["work_behavior"]

            task_prefix = random.choice(profile['task_name_prefix'])
            task_jargon = random.choice(domain_data['task_jargon'])
            specific_entity = str(random.randint(100, 9999)) if "#" in task_jargon else fake.name()
            task_name = f"{task_prefix} {task_jargon}{specific_entity}"
            description = fake.text(max_nb_chars=random.randint(*profile["description_length"]))

            # MODIFICARE CHEIE: Generăm datele temporale pentru a se potrivi cu statusul țintă
            creation_date, due_date = generate_realistic_dates_for_status(
                target_status,
                behavior,
                domain_data["average_duration_range"]
            )

            status = target_status
            status_counts[status] += 1

            cur.execute(
                "INSERT INTO task (creation_date, description, due_date, name, status, board_id) VALUES (%s, %s, %s, %s, %s, %s);",
                (creation_date, description, due_date, task_name, status, board_id)
            )

            if (i + 1) % 500 == 0:
                print(f"  ... {i + 1}/{NUM_TASKS} task-uri generate.")

        conn.commit()
        cur.close()
        print("\nGenerarea datelor a fost finalizata cu succes!")
        print(f"Distribuția finală a statusurilor: {status_counts}")

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Eroare: {error}")
        if conn is not None:
            conn.rollback()
    finally:
        if conn is not None:
            conn.close()
            print("Conexiunea la baza de date a fost inchisa.")


if __name__ == "__main__":
    generate_data()