import tkinter as tk

class BattleshipGUI:
    def __init__(self, env, agent=None):
        self.env = env
        self.agent = agent
        self.window = tk.Tk()
        self.window.title("Battleship Game")
        self.buttons = []
        self.state, _ = self.env.reset()
        self.done = False
        self.vs_agent = False
        self.alternate_turns = False
        self.player_turn = True
        self.destroyed_cells = set()

        tk.Label(self.window, text="Player Name:").grid(row=self.env.grid_size+3, column=0, columnspan=2)
        self.player_name = tk.Entry(self.window)
        self.player_name.grid(row=self.env.grid_size+3, column=2, columnspan=self.env.grid_size-2)
        self.player_name.insert(0, "Player")

        for r in range(self.env.grid_size):
            self.window.grid_rowconfigure(r, weight=1, uniform="grid")
            row = []
            for c in range(self.env.grid_size):
                self.window.grid_columnconfigure(c, weight=1, uniform="grid")
                btn = tk.Button(self.window, text=" ", width=4, height=2, font=("Arial", 10),
                                command=lambda r=r, c=c: self.handle_click(r, c))
                btn.grid(row=r, column=c, sticky="nsew")
                row.append(btn)
            self.buttons.append(row)

        self.status_label = tk.Label(self.window, text="Your move")
        self.status_label.grid(row=self.env.grid_size, column=0, columnspan=self.env.grid_size)

        self.play_agent_button = tk.Button(self.window, text="Play Against Agent", command=self.play_vs_agent)
        self.play_agent_button.grid(row=self.env.grid_size+1, column=0, columnspan=self.env.grid_size//3)

        self.play_turns_button = tk.Button(self.window, text="Alternate Turns With Agent", command=self.play_alternative_turns)
        self.play_turns_button.grid(row=self.env.grid_size+1, column=self.env.grid_size//3, columnspan=self.env.grid_size//3)

        self.reset_button = tk.Button(self.window, text="Reset Game", command=self.reset_game)
        self.reset_button.grid(row=self.env.grid_size+1, column=2*self.env.grid_size//3, columnspan=self.env.grid_size//3)

        self.restart_button = tk.Button(self.window, text="Restart Game", command=self.reset_game)
        self.restart_button.grid(row=self.env.grid_size+2, column=0, columnspan=self.env.grid_size)
        self.restart_button.grid_remove()

        self.window.mainloop()

    def handle_click(self, row, col):
        if self.done or (self.vs_agent and not self.alternate_turns) or (self.vs_agent and self.alternate_turns and not self.player_turn):
            return
        action = row * self.env.grid_size + col
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.state = obs
        self.update_buttons(obs, info)
        if reward == -0.2:
            self.status_label.config(text="Cell already selected!")
            return
        elif reward >= 1.0:
            self.status_label.config(text="Hit! Continue playing")
            if info.get("sunk_ship"):
                self.status_label.config(text=f"You sunk a ship!")
        elif reward <= -0.05:
            self.status_label.config(text="Miss")
        if terminated:
            player_name = self.player_name.get() or "Player"
            self.status_label.config(text=f"{player_name} Wins! All ships destroyed!")
            self.done = True
            self.restart_button.grid()
            return
        elif truncated:
            self.status_label.config(text="Game truncated: Too many steps!")
            self.done = True
            self.restart_button.grid()
            return
        if self.vs_agent and self.alternate_turns and reward < 1.0:
            self.player_turn = False
            self.window.after(500, self.agent_step)

    def update_buttons(self, obs, info=None):
        for r in range(self.env.grid_size):
            for c in range(self.env.grid_size):
                if info and info.get("sunk_ship") and (r, c) in info["sunk_ship"]:
                    self.destroyed_cells.add((r, c))
                if (r, c) in self.destroyed_cells:
                    self.buttons[r][c].config(bg="green", text="X")
                else:
                    val = obs[r, c]
                    if val == 1:
                        self.buttons[r][c].config(bg="red", text="X")
                    elif val == 0:
                        self.buttons[r][c].config(bg="blue", text="O")
                    else:
                        self.buttons[r][c].config(bg="SystemButtonFace", text=" ")

    def reset_game(self):
        self.state, _ = self.env.reset()
        self.done = False
        self.player_turn = True
        self.destroyed_cells.clear()
        self.status_label.config(text="Your move")
        self.update_buttons(self.state)
        self.restart_button.grid_remove()
        for r in range(self.env.grid_size):
            for c in range(self.env.grid_size):
                self.buttons[r][c].config(width=4)

    def play_vs_agent(self):
        if not self.agent:
            self.status_label.config(text="No agent loaded")
            return
        self.vs_agent = True
        self.alternate_turns = False
        self.reset_game()
        self.status_label.config(text="Agent is playing...")
        self.window.after(500, self.agent_step)

    def play_alternative_turns(self):
        if not self.agent:
            self.status_label.config(text="No agent loaded")
            return
        self.vs_agent = True
        self.alternate_turns = True
        self.reset_game()
        self.status_label.config(text="Your move - alternating turns")

    def agent_step(self):
        if self.done:
            return
        action = self.agent.choose_action(self.state)
        self.state, reward, terminated, truncated, info = self.env.step(action)
        self.update_buttons(self.state, info)
        if info.get("sunk_ship"):
            self.status_label.config(text="Agent sunk a ship!")
        if terminated:
            self.status_label.config(text="Agent Wins! All ships destroyed!")
            self.done = True
            self.restart_button.grid()
            return
        elif truncated:
            self.status_label.config(text="Agent failed: Too many steps!")
            self.done = True
            self.restart_button.grid()
            return
        if self.alternate_turns:
            if reward < 1.0:
                self.player_turn = True
                self.status_label.config(text="Your move")
            else:
                self.window.after(500, self.agent_step)
        else:
            self.window.after(500, self.agent_step)