import csv
from datetime import datetime, date, timedelta
import os
import sys
import re

class StudySession:
    def __init__(self, subject: str, date: str, hours: float, notes: str):
        self._subject = subject
        self._date = date
        self._hours = hours
        self._notes = notes
    @property
    def subject(self) -> str:
        return self._subject
    @property
    def date(self) -> str:
        return self._date
    @property
    def hours(self) -> float:
        return self._hours
    @property
    def notes(self) -> str:
        return self._notes
    def __str__(self) -> str:
        return f"[{self._date}] {self._subject}: {self._hours} hours - {self._notes}"
    def to_dict(self) -> dict:
        return {
            "subject": self._subject,
            "date": self._date,
            "hours": self._hours,
            "notes": self._notes
        }
    @classmethod
    def from_dict(cls, row: dict) -> "StudySession":
        return cls(row["subject"].strip(), row["date"].strip(), float(row["hours"]), row["notes"].strip())
    
def validate_date(date_str: str) -> bool:
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str.strip()):
        return False
    try:
        datetime.strptime(date_str.strip(), "%Y-%m-%d")
        return True
    except ValueError:
        return False
def validate_hours(hours: str) -> float:
    
        try:
            hoursF = float(hours)
        except ValueError :
            raise ValueError("Invalid input for hours. Please enter a valid number.")
        if hoursF <= 0:
            raise ValueError("Hours cannot be zero or negative.")
        return hoursF
def calculate_total_hours(sessions):
    total = 0.0
    for session in sessions:
        total += session.hours
    return total
def filter_sessions_by_subject(sessions, subject):
    return [session for session in sessions if session.subject.lower() == subject.lower()]
def load_sessions(filepath: str) -> list:
    sessions = []
    if not os.path.exists(filepath):
        return sessions
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sessions.append(StudySession.from_dict(row))
    return sessions
def save_sessions(sessions: list, filepath: str):
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subject", "date", "hours", "notes"])
        writer.writeheader()
        for session in sessions:
            writer.writerow(session.to_dict())
def display_menu():
    print("=== Study Tracker ===")
    print("1. Log a study session")
    print("2. View sessions")
    print("3. Weekly stats")
    print("4. Export report")
    print("5. Exit")

def log_session(sessions):
    while True:
        date = input("Enter the date (YYYY-MM-DD): ")
        if validate_date(date):
            break   
        print("Invalid date format. Please try again.")
    while True:
        subject = input("Enter the subject: ")
        if subject.strip():
            break
        print("Subject cannot be empty.")
    while True:
        hours = input("Enter hours studied: ")
        try:
            hoursF = validate_hours(hours)
            break
        except ValueError as e:
            print(e)
    notes = input("Enter any notes (optional): ")
    sessions.append(StudySession(subject, date, hoursF, notes))
    save_sessions(sessions, "study_sessions.csv")
    print("Study session logged successfully!")
    return sessions
def view_sessions(sessions: list):
    if not sessions:
        print("No sessions logged yet.")
        return
    print("1. All")
    print("2. Filter by subject")
    while True:
        choice = input("Choose an option: ")
        if choice in ["1", "2"]:
            break
        print("Please enter 1 or 2.")
    if choice == "2":
        while True:
            subject = input("Enter subject: ")
            if subject.strip():
                break
            print("Subject cannot be empty.")
        results = filter_sessions_by_subject(sessions, subject)
        if not results:
            print("No sessions found for that subject.")
            return
        for i, session in enumerate(results, 1):
            print(f"{i}. {session}")
    else:
        for i, session in enumerate(sessions, 1):
            print(f"{i}. {session}")
def weekly_stats(sessions):
    today = date.today()
    weekAgo = today - timedelta(days=7)
    recent_sessions = [session for session in sessions if datetime.strptime(session.date.strip(), "%Y-%m-%d").date() >= weekAgo]
    if not recent_sessions:
        print("No sessions logged this week.")
        return
    subjs = set(session.subject.strip() for session in recent_sessions)
    print("=== This Week's Stats ===")
    for subject in subjs:
        subject_sessions = filter_sessions_by_subject(recent_sessions, subject)
        total = calculate_total_hours(subject_sessions)
        print(f"{subject:<15} — {total} hrs")
    print("─" * 30)
    print(f"{'Total':<15} — {calculate_total_hours(recent_sessions)} hrs")
    
    streak = 0
    check_date = today
    while True:
        day_sessions = [s for s in sessions if datetime.strptime(s.date.strip(), "%Y-%m-%d").date() == check_date]
        if not day_sessions:
            break
        streak += 1
        check_date -= timedelta(days=1)
    
    print(f"Current streak: {streak} days")
def export_report(sessions: list):
    if not sessions:
        print("Nothing to export yet.")
        return
    with open("study_report.txt", "w") as f:
        today = date.today()
        f.write("=" * 50 + "\n")
        f.write(f"STUDY REPORT — Generated on {today}\n")
        f.write("=" * 50 + "\n")
        subjs1 = set(session.subject for session in sessions)
        for subject in subjs1:
            f.write(f"\n{subject.upper()}\n")
            subject_sessions = filter_sessions_by_subject(sessions, subject)
            for session in subject_sessions:
                f.write(f"  {session.date} | {session.hours} hrs | {session.notes}\n")
            f.write(f"  Subject total: {calculate_total_hours(subject_sessions)} hrs\n")
        f.write("\n" + "=" * 50 + "\n")
        f.write(f"GRAND TOTAL: {calculate_total_hours(sessions)} hrs across {len(sessions)} sessions\n")
    print("Report saved to study_report.txt.")
def main():
    if len(sys.argv) == 2 and sys.argv[1] == "--report":
        sessions = load_sessions("study_sessions.csv")
        export_report(sessions)
        return
    elif len(sys.argv) > 1:
        print("Unknown argument. Use --report or run with no arguments.")
        return
    
    sessions = load_sessions("study_sessions.csv")
    while True:
        display_menu()
        choice = input("Choose an option: ")
        if choice == "1":
            sessions = log_session(sessions)
        elif choice == "2":
            view_sessions(sessions)
        elif choice == "3":
            weekly_stats(sessions)
        elif choice == "4":
            export_report(sessions)
        elif choice == "5":
            print("Goodbye!")
            break
        else:
            print("Invalid option. Please try again.")


if __name__ == "__main__":
    main()
#code is edited on github

                    

        


    

                        
               
        
