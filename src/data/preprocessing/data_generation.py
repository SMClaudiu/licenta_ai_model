import psycopg2
import random
import bcrypt
from faker import Faker
from datetime import datetime, timedelta

from torch.distributed import init_process_group

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

#Justificare pentru durata medie si cat de urgent este
CORPORATE_DOMAINS = {
    "it_dev": {
        "board_names": ["Backend Refactor Sprint", "API Gateway Migration", "Q3 Security Audit", "Production Hotfixes",
                        "UI/UX Overhaul"],
        "task_jargon": ["deploy to Staging", "fix bug #", "code review PR #", "optimize DB query for ",
                        "investigate latency in ", "refactor service "],
        "average_duration_range": (1,14),
        "urgency_multiplier": 1.3
    },
    "marketing": {
        "board_names": ["Q4 Social Media Campaign", "New Product Launch", "Content Calendar - Blog",
                        "SEO & SEM Strategy", "Brand Awareness Initiative"],
        "task_jargon": ["draft press release for ", "analyze campaign results of ", "create new ad copy for ",
                        "schedule social media posts for ", "A/B test landing page ", "update content brief for "],
        "average_duration_range": (3,21),
        "urgency_multiplier": 0.9
    },
    "sales": {
        "board_names": ["Q3 Lead Pipeline", "Key Account - Acme Corp", "Cold Outreach Strategy",
                        "Sales Team Weekly Sync", "New Market Expansion"],
        "task_jargon": ["follow up with client ", "prepare demo for prospect ", "update CRM with notes for ",
                        "log call with ", "research potential leads in ", "finalize contract for "],
        "average_duration_range": (1,7), #vorbim despre lucruri ce nu pot fi decalate de factori externi precum procurement(PR2PO)
        "urgency_multiplier": 1.2
    },
    "hr": {
        "board_names": ["New Hire Onboarding", "Performance Review Cycle", "Company Culture Initiative",
                        "Recruitment - Senior Dev", "Benefits Package Review"],
        "task_jargon": ["schedule interview for ", "process payroll changes for ", "prepare benefits package for ",
                        "draft new company policy on ", "conduct exit interview with ", "organize team building event "],
        "average_duration_range": (2,14),
        "urgency_multiplier": 0.8
    },
    "finance": {
        "board_names": ["End-of-Month Closing", "Budget Planning 2024", "Expense Report Approvals", "Annual Audit Prep",
                        "Investor Relations Deck"],
        "task_jargon": ["reconcile accounts for ", "process invoice #", "generate P&L report for ",
                        "approve expense report for ", "analyze financial variance for ", "forecast cash flow for "],
        "average_duration_range": (1,10),
        "urgency_multiplier": 1.1
    }
}

USER_PROFILES = {
    "beginner": {
        "weight": 35,
        "dashboards_per_client": (1, 1),
        "boards_per_dashboard": (1, 2),
        "task_name_prefix": ["Do", "Check", "Buy", "Call"],  # Verbe simple
        "description_length": (5, 50),

        #Cu cat e mai aproape de 1 este predispus de a completa acel profil
        "work_behavior": {
            "multitasking_tendency": 0.2,
            "procrastination_tendency": 0.4,
            "weekend_work_probability": 0.05,
            "overtime_probability": 0.1,
            "task_completion_consistency": 0.6,
            "urgent_chance": 0.1
        }
    },
    "medium": {
        "weight": 50,
        "dashboards_per_client": (1, 3),
        "boards_per_dashboard": (2, 5),
        "task_name_prefix": ["Review", "Draft", "Implement", "Prepare", "Finalize"],
        "description_length": (50, 250),
         "work_behavior": {
            "multitasking_tendency": 0.6,
            "procrastination_tendency": 0.2,
            "weekend_work_probability": 0.15,
            "overtime_probability": 0.3,
            "task_completion_consistency": 0.8,
            "urgent_chance": 0.25
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
            "task_completion_consistency": 0.95,  # Very consistent
            "urgent_chance": 0.5
        }
    }
}


def get_weighted_profile_name():

    profiles = list(USER_PROFILES.keys())
    weights = [USER_PROFILES[p]['weight'] for p in profiles]
    return random.choices(profiles, weights, k=1)[0]

def calculate_task_status(task_data, behavior, current_time):

    creation_date = task_data["creation_date"]
    due_date = task_data["due_date"]

    #Timp de cand a fost creat
    task_age_hours = (current_time - creation_date).total_seconds() / 3600
    task_age_days = task_age_hours / 24

    #Timp alocat
    hours_due = (due_date - creation_date).total_seconds() / 3600
    days_due = hours_due /24

    if days_due > 0:
        time_passed = task_age_days / days_due
    else: time_passed = 1.0
    percentage_tp = min(time_passed, 1.0)

    is_overdue = hours_due < 0


    priority_keywords = any(keyword in task_data["name"].lower() for keyword in
                            ['urgent', 'critical', 'blocker', 'asap', 'emergency', 'must', 'Live'])

    #Tipare anagajati
    multitasking = behavior['multitasking_tendency']
    procrastination = behavior['procrastination_tendency']
    weekend_work = behavior['weekend_work_probability']
    overtime = behavior['overtime_probability']
    consistency = behavior['task_completion_consistency']

    #Tipare de completare a task-urilor
    completion_probability = 0.0
    inprogress_probability = 0.0

    # Completion Probability Formula
    #
    # w1 * (- P) + w2 * CC + w3 * WW + w4 * O + w5 * MT
    #
    # P - Procrastination
    # C - Consistency
    # WW - Working Weekends
    # O - Overtime
    # MT - Multitasking Tendency
    #
    # from 0 to 1 importance
    # w - importanta relativa a fiecarei trasaturi comportamentale(greutatea)


    if task_age_hours < 1:
        return 0 #Pending

    #II - Completion probability de vazut daca 0.4 modifica cu ceva
    if is_overdue:
        completion_probability = 0.3 - procrastination * 0.2 + consistency * 0.6 + weekend_work * 0.1 + overtime * 0.2
    else:
        completion_probability = percentage_tp ** 2 * consistency - multitasking * 0.1

    if priority_keywords:
        completion_probability += 0.3

    #I - In-progress probability
    procrastination_treshold = procrastination * 0.5

    if percentage_tp > procrastination_treshold:
        max_effort = 1 - abs(percentage_tp - 0.5 ) * 2     # parabola de efort maxim a unui task
        inprogress_probability = (1 - procrastination) * 0.5 + multitasking * 0.4 + max_effort * 0.3

    if priority_keywords and not is_overdue:
        inprogress_probability += 0.2

    #Probability proofing
    completion_probability = max(0.0, min(completion_probability, 1))
    inprogress_probability = max(0.0, min(inprogress_probability, 1))

    if completion_probability + inprogress_probability > 1:
        inprogress_probability = 1 - completion_probability

    rand = random.random()
    if rand <completion_probability:
        return 2 # Status code for "Completed"
    elif rand < completion_probability + inprogress_probability:
        return 1 # Status code for "In progress"
    else:
        return 0 # Status code for "Pending"


def generate_data():
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        print("Conectat la baza de date.")

        print("Curatarea tabelelor existente...")
        cur.execute("TRUNCATE TABLE task, board, dashboard, client RESTART IDENTITY CASCADE;")
        conn.commit()

        clients_data = []
        boards_data = []

        print(f"Generare {NUM_CLIENTS} clienti...")
        for _ in range(NUM_CLIENTS):
            profile_name = get_weighted_profile_name()
            # Asigneaza un domeniu corporate aleatoriu fiecarui client
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
            clients_data.append({"client_id": client_id, "profile": profile_name, "domain": domain_name})

        print("Generare dashboards si boards...")
        for client in clients_data:
            profile = USER_PROFILES[client["profile"]]
            domain_data = CORPORATE_DOMAINS[client["domain"]]

            num_dashboards = random.randint(*profile["dashboards_per_client"])
            for _ in range(num_dashboards):
                # Numele dashboard-ului este un nume de board din domeniul clientului
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
                    # Stocam si domeniul pentru a-l folosi la generarea task-urilor
                    boards_data.append({"board_id": board_id, "profile": client["profile"], "domain": client["domain"]})

        print(f"Generare {NUM_TASKS} task-uri...")
        for i in range(NUM_TASKS):
            board_info = random.choice(boards_data)
            board_id = board_info["board_id"]

            profile_name = board_info["profile"]
            domain_name = board_info["domain"]

            profile = USER_PROFILES[profile_name]
            domain_data = CORPORATE_DOMAINS[domain_name]

            task_prefix = random.choice(profile['task_name_prefix'])
            task_jargon = random.choice(domain_data['task_jargon'])

            behavior = profile["work_behavior"]

            # Adauga un element specific (ex: nume, numar) pentru varietate
            specific_entity = ""
            if "#" in task_jargon:
                specific_entity = str(random.randint(100, 9999))
            else:
                specific_entity = fake.company() if random.random() > 0.5 else fake.name()

            task_name = f"{task_prefix} {task_jargon}{specific_entity}"

            description = fake.text(max_nb_chars=random.randint(*profile["description_length"]))
            status = random.choice([0, 1, 2])

            creation_date = datetime.now() - timedelta(days=random.randint(1, 45), hours=random.randint(0, 23))

            min_duration, max_duration = domain_data["average_duration_range"]

            days_to_complete = random.randint(1, 30)
            if random.random() < behavior["urgent_chance"]:

                days_to_complete = random.uniform(0.5, 3)

            due_date = creation_date + timedelta(days=days_to_complete)

            task_data_for_calc = {
                "name": task_name,
                "creation_date": creation_date,
                "due_date": due_date,
            }

            status = calculate_task_status(task_data_for_calc, behavior, datetime.now())

            cur.execute(
                """
                INSERT INTO task (creation_date, description, due_date, name, status, board_id)
                VALUES (%s, %s, %s, %s, %s, %s);
                """,
                (creation_date, description, due_date, task_name, status, board_id)
            )

            if (i + 1) % 500 == 0:
                print(f"  ... {i + 1}/{NUM_TASKS} task-uri generate.")

        conn.commit()
        cur.close()
        print("\nGenerarea datelor a fost finalizata cu succes!")

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