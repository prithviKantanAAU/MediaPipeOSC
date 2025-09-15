import tkinter as tk
from tkinter import ttk, messagebox
import datetime
import json
import os
from tkcalendar import DateEntry

class AcademicUtilitySuite:
    def __init__(self, root):
        self.root = root
        self.root.title("Academic Task Manager")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')
        
        # Data storage
        self.tasks_file = "academic_tasks.json"
        self.load_data()
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create the task manager tab
        self.create_task_manager_tab()
        
        # Save data on close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def load_data(self):
        """Load saved tasks from JSON file"""
        try:
            with open(self.tasks_file, 'r') as f:
                self.tasks = json.load(f)
        except FileNotFoundError:
            self.tasks = []
            
    def save_data(self):
        """Save tasks to JSON file"""
        with open(self.tasks_file, 'w') as f:
            json.dump(self.tasks, f, indent=4)
    
    def on_closing(self):
        """Handle application closing"""
        self.save_data()
        self.root.destroy()
    
    def create_task_manager_tab(self):
        """Task and assignment manager"""
        task_frame = ttk.Frame(self.notebook)
        self.notebook.add(task_frame, text="Task Manager")
        
        tk.Label(task_frame, text="Academic Task Manager", font=("Arial", 14, "bold")).pack(pady=10)
        
        # Add task frame
        add_frame = tk.Frame(task_frame)
        add_frame.pack(pady=10, fill='x', padx=20)
        
        tk.Label(add_frame, text="Task:").grid(row=0, column=0, sticky='w', pady=5)
        self.task_entry = tk.Entry(add_frame, width=30)
        self.task_entry.grid(row=0, column=1, pady=5, padx=5)
        
        tk.Label(add_frame, text="Due Date:").grid(row=0, column=2, sticky='w', pady=5, padx=(20,0))
        self.due_date_entry = DateEntry(add_frame, width=12, date_pattern='dd-mm-yy',
                                        borderwidth=2)
        self.due_date_entry.grid(row=0, column=3, pady=5, padx=5)
        self.due_date_entry.delete(0, "end")

        tk.Label(add_frame, text="Priority:").grid(row=1, column=0, sticky='w', pady=5)
        self.priority_var = tk.StringVar(value="Medium")
        priority_combo = ttk.Combobox(add_frame, textvariable=self.priority_var, 
                                     values=["High", "Medium", "Low"], width=10)
        priority_combo.grid(row=1, column=1, pady=5, padx=5, sticky='w')
        
        tk.Button(add_frame, text="Add Task", command=self.add_task).grid(row=1, column=2, columnspan=2, pady=5, padx=20)
        
        # Task list
        list_frame = tk.Frame(task_frame)
        list_frame.pack(pady=10, fill='both', expand=True, padx=20)
        
        columns = ('Task', 'Due Date', 'Days Remaining', 'Priority', 'Status')
        self.task_tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=12)
        
        self.task_tree.heading('Task', text='Task')
        self.task_tree.column('Task', width=250)
        
        self.task_tree.heading('Due Date', text='Due Date')
        self.task_tree.column('Due Date', width=100, anchor='center')
        
        self.task_tree.heading('Days Remaining', text='Days Remaining')
        self.task_tree.column('Days Remaining', width=120, anchor='center')
        
        self.task_tree.heading('Priority', text='Priority')
        self.task_tree.column('Priority', width=80, anchor='center')
        
        self.task_tree.heading('Status', text='Status')
        self.task_tree.column('Status', width=80, anchor='center')
        
        scrollbar = ttk.Scrollbar(list_frame, orient='vertical', command=self.task_tree.yview)
        self.task_tree.configure(yscrollcommand=scrollbar.set)
        
        self.task_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        self.task_tree.bind("<Delete>", self.delete_task_event)
        
        # Task controls
        control_frame = tk.Frame(task_frame)
        control_frame.pack(pady=10)
        
        tk.Button(control_frame, text="Complete Task", command=self.complete_task).pack(side='left', padx=5)
        tk.Button(control_frame, text="Delete Task", command=self.delete_task).pack(side='left', padx=5)
        
        self.refresh_tasks()
    
    def add_task(self):
        """Add a new task"""
        task = self.task_entry.get().strip()
        due_date = self.due_date_entry.get().strip()
        priority = self.priority_var.get()
        
        if not task:
            messagebox.showwarning("Invalid Task", "Please enter a task description")
            return
        
        new_task = {
            "task": task,
            "due_date": due_date,
            "priority": priority,
            "status": "Pending",
            "created": datetime.datetime.now().isoformat()
        }
        
        self.tasks.append(new_task)
        self.task_entry.delete(0, tk.END)
        self.due_date_entry.delete(0, tk.END)
        self.refresh_tasks()
    
    def complete_task(self):
        """Mark selected task as complete"""
        selected_items = self.task_tree.selection()
        if not selected_items:
            messagebox.showwarning("No Selection", "Please select a task to complete")
            return
        
        for item_id in selected_items:
            selected_item_values = self.task_tree.item(item_id)['values']
            task_text = selected_item_values[0]
            due_date = selected_item_values[1]

            for task in self.tasks:
                if task['task'] == task_text and task['due_date'] == due_date:
                    task['status'] = 'Completed'
                    break
        
        self.refresh_tasks()
    
    def delete_task_event(self, event):
        """Handles the delete key press event."""
        self.delete_task()

    def delete_task(self):
        """Delete selected task(s) without a confirmation prompt."""
        selected_items = self.task_tree.selection()
        if not selected_items:
            return # Do nothing if no task is selected
        
        tasks_to_delete = []
        for item_id in selected_items:
            item_values = self.task_tree.item(item_id)['values']
            tasks_to_delete.append({'task': item_values[0], 'due_date': item_values[1]})
        
        # Filter out the selected tasks
        self.tasks = [
            t for t in self.tasks 
            if not any(
                d['task'] == t['task'] and d['due_date'] == t['due_date'] 
                for d in tasks_to_delete
            )
        ]
        self.refresh_tasks()

    def refresh_tasks(self):
        """Refresh task list display with sorting and days remaining"""
        for item in self.task_tree.get_children():
            self.task_tree.delete(item)
        
        today = datetime.date.today()
        priority_order = {"High": 0, "Medium": 1, "Low": 2}

        def get_days_remaining(due_date_str):
            if not due_date_str:
                return float('inf')
            try:
                due_date = datetime.datetime.strptime(due_date_str, "%d-%m-%y").date()
                return (due_date - today).days
            except ValueError:
                return float('inf')
        
        # **CHANGE:** The sorting key now only uses get_days_remaining.
        sorted_tasks = sorted(self.tasks, key=lambda x: get_days_remaining(x['due_date']))
        
        for task in sorted_tasks:
            days_remaining = get_days_remaining(task['due_date'])
            
            if days_remaining == float('inf'):
                days_display = "N/A"
            elif days_remaining < 0:
                days_display = f"Overdue by {-days_remaining} days"
            elif days_remaining == 0:
                days_display = "Today"
            else:
                days_display = f"{days_remaining} days"

            tags = []
            if task['status'] == 'Completed':
                tags.append('completed')
            elif task['priority'] == 'High' and task['status'] == 'Pending':
                tags.append('high_priority')
            if days_remaining < 0 and task['status'] == 'Pending':
                tags.append('overdue')

            self.task_tree.insert('', 'end', values=(
                task['task'], 
                task['due_date'], 
                days_display,
                task['priority'], 
                task['status']
            ), tags=tags)
        
        self.task_tree.tag_configure('completed', background='#c8e6c9')
        self.task_tree.tag_configure('high_priority', background='#ffcdd2')
        self.task_tree.tag_configure('overdue', foreground='red')

if __name__ == "__main__":
    root = tk.Tk()
    app = AcademicUtilitySuite(root)
    root.mainloop()