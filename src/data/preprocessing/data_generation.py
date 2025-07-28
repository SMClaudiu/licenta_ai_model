import psycopg2
import random
import bcrypt
from faker import Faker
from datetime import datetime, timedelta

# --- CONFIGURARE ---
DB_CONFIG = {
    "dbname": "licenta_db",
    "user": "postgres",
    "password": "101102",
    "host": "localhost",
    "port": "5432"
}

# Constante pentru generare
NUM_CLIENTS = 200
NUM_TASKS = 5000

# Initializare Faker pentru date false
fake = Faker()

# --- NOU: DEFINIREA DOMENIILOR CORPORATE ---
# Aceasta structura defineste "ce" se face in fiecare departament.
CORPORATE_DOMAINS = {
    "it_dev": {
        "board_names": ["Backend Refactor Sprint", "API Gateway Migration", "Q3 Security Audit", "Production Hotfixes",
                        "UI/UX Overhaul"],
        "task_jargon": ["deploy to Staging", "fix bug #", "code review PR #", "optimize DB query for ",
                        "investigate latency in ", "refactor service "]
    },
    "marketing": {
        "board_names": ["Q4 Social Media Campaign", "New Product Launch", "Content Calendar - Blog",
                        "SEO & SEM Strategy", "Brand Awareness Initiative"],
        "task_jargon": ["draft press release for ", "analyze campaign results of ", "create new ad copy for ",
                        "schedule social media posts for ", "A/B test landing page ", "update content brief for "]
    },
    "sales": {
        "board_names": ["Q3 Lead Pipeline", "Key Account - Acme Corp", "Cold Outreach Strategy",
                        "Sales Team Weekly Sync", "New Market Expansion"],
        "task_jargon": ["follow up with client ", "prepare demo for prospect ", "update CRM with notes for ",
                        "log call with ", "research potential leads in ", "finalize contract for "]
    },
    "hr": {
        "board_names": ["New Hire Onboarding", "Performance Review Cycle", "Company Culture Initiative",
                        "Recruitment - Senior Dev", "Benefits Package Review"],
        "task_jargon": ["schedule interview for ", "process payroll changes for ", "prepare benefits package for ",
                        "draft new company policy on ", "conduct exit interview with ", "organize team building event "]
    },
    "finance": {
        "board_names": ["End-of-Month Closing", "Budget Planning 2024", "Expense Report Approvals", "Annual Audit Prep",
                        "Investor Relations Deck"],
        "task_jargon": ["reconcile accounts for ", "process invoice #", "generate P&L report for ",
                        "approve expense report for ", "analyze financial variance for ", "forecast cash flow for "]
    }
}

# --- DEFINIREA ARHETIPURILOR (Nivel de experienta) ---
# Aceasta structura defineste "cum" lucreaza un utilizator.
USER_PROFILES = {
    "beginner": {
        "weight": 35,
        "dashboards_per_client": (1, 1),
        "boards_per_dashboard": (1, 2),
        "task_name_prefix": ["Do", "Check", "Buy", "Call"],  # Verbe simple
        "description_length": (5, 50),
        "overdue_chance": 0.4,
        "urgent_chance": 0.05,
        "weekend_creation_chance": 0.05,
    },
    "medium": {
        "weight": 50,
        "dashboards_per_client": (1, 3),
        "boards_per_dashboard": (2, 5),
        "task_name_prefix": ["Review", "Draft", "Implement", "Prepare", "Finalize"],  # Verbe orientate spre proces
        "description_length": (50, 250),
        "overdue_chance": 0.15,
        "urgent_chance": 0.2,
        "weekend_creation_chance": 0.15,
    },
    "advanced": {
        "weight": 15,
        "dashboards_per_client": (3, 7),
        "boards_per_dashboard": (5, 10),
        "task_name_prefix": ["URGENT: Fix", "BLOCKER:", "CRITICAL: Deploy", "Investigate", "Optimize"],
        # Orientat spre prioritate si impact
        "description_length": (250, 1000),
        "overdue_chance": 0.05,
        "urgent_chance": 0.6,
        "weekend_creation_chance": 0.4,
    }
}


def get_weighted_profile_name():
    """Alege un profil de experienta pe baza distributiei de probabilitate."""
    profiles = list(USER_PROFILES.keys())
    weights = [USER_PROFILES[p]['weight'] for p in profiles]
    return random.choices(profiles, weights, k=1)[0]


def generate_data():
    """Functia principala care genereaza si insereaza datele."""
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

        # --- 1. GENERARE CLIENTI cu PROFIL si DOMENIU ---
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

        # --- 2 & 3. GENERARE DASHBOARDS & BOARDS pe baza domeniului ---
        print("Generare dashboards si boards...")
        for client in clients_data:
            profile = USER_PROFILES[client["profile"]]
            domain_data = CORPORATE_DOMAINS[client["domain"]]  # Obtine datele specifice domeniului

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

        # --- 4. GENERARE TASKS pe baza PROFILULUI si DOMENIULUI ---
        print(f"Generare {NUM_TASKS} task-uri...")
        for i in range(NUM_TASKS):
            board_info = random.choice(boards_data)
            board_id = board_info["board_id"]

            profile_name = board_info["profile"]
            domain_name = board_info["domain"]

            profile = USER_PROFILES[profile_name]
            domain_data = CORPORATE_DOMAINS[domain_name]

            # Combina prefixul din profil cu jargonul din domeniu pentru un nume de task realist
            task_prefix = random.choice(profile['task_name_prefix'])
            task_jargon = random.choice(domain_data['task_jargon'])

            # Adauga un element specific (ex: nume, numar) pentru varietate
            specific_entity = ""
            if "#" in task_jargon:
                specific_entity = str(random.randint(100, 9999))
            else:
                specific_entity = fake.company() if random.random() > 0.5 else fake.name()

            task_name = f"{task_prefix} {task_jargon}{specific_entity}"

            description = fake.text(max_nb_chars=random.randint(*profile["description_length"]))
            status = random.choice([0, 1, 2])

            creation_date = datetime.now() - timedelta(days=random.randint(1, 365), hours=random.randint(0, 23))

            # Generare termen limita
            days_to_complete = random.randint(1, 30)
            if random.random() < profile["urgent_chance"]:
                days_to_complete = random.randint(1, 3)
            due_date = creation_date + timedelta(days=days_to_complete)

            # Simulare status "overdue"
            if status != 2 and random.random() < profile["overdue_chance"]:
                due_date = datetime.now() - timedelta(days=random.randint(1, 10))

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