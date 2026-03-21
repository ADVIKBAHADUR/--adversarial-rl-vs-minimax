"""Simple Pygame GUI — paper-ready, big fonts, clean design."""

import argparse
import sys
import numpy as np

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False

from .games import TicTacToe, Connect4
from .agents import DefaultAgent, MinimaxAgent, QLearningAgent, DQNAgent, HumanAgent


# ── Colours ───────────────────────────────────────────────────────────────────

BG_COLOUR      = (30, 30, 40)
GRID_COLOUR    = (70, 70, 90)
TEXT_COLOUR    = (220, 220, 230)
P1_COLOUR      = (46, 204, 113)   # Green for X / Player 1
P2_COLOUR      = (231, 76, 60)    # Red for O / Player 2
EMPTY_COLOUR   = (50, 50, 65)
HIGHLIGHT      = (52, 152, 219)   # Blue highlight
PANEL_COLOUR   = (40, 40, 55)


# ── GUI class ─────────────────────────────────────────────────────────────────

class GameGUI:
    """Pygame-based game GUI with big, clear visuals for paper screenshots."""

    CELL_SIZE = 100
    PADDING = 40
    INFO_WIDTH = 280
    FONT_TITLE = 36
    FONT_INFO = 22
    FONT_CELL = 48
    FONT_STATUS = 28

    def __init__(self, game, agent1, agent2):
        if not HAS_PYGAME:
            print("Pygame not installed. Install with: pip install pygame")
            sys.exit(1)

        self.game = game
        self.agents = {1: agent1, -1: agent2}
        self.state = game.reset()
        self.player = 1
        self.done = False
        self.winner = 0
        self.move_history = []
        self.status_msg = f"{agent1.name}'s turn (X)"

        # GUI sizing
        rows, cols = game.state_shape
        self.board_w = cols * self.CELL_SIZE
        self.board_h = rows * self.CELL_SIZE
        self.width = self.board_w + self.INFO_WIDTH + self.PADDING * 3
        self.height = self.board_h + self.PADDING * 3 + 60  # +60 for title

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(f"{game.name} — Adversarial RL vs Minimax")

        self.font_title = pygame.font.SysFont("Arial", self.FONT_TITLE, bold=True)
        self.font_info = pygame.font.SysFont("Arial", self.FONT_INFO)
        self.font_cell = pygame.font.SysFont("Arial", self.FONT_CELL, bold=True)
        self.font_status = pygame.font.SysFont("Arial", self.FONT_STATUS, bold=True)

        self.is_human_turn = isinstance(self.agents[1], HumanAgent)
        self.hover_col = -1

    def run(self):
        """Main game loop."""
        clock = pygame.time.Clock()
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEMOTION:
                    self.hover_col = self._get_col(event.pos)
                elif event.type == pygame.MOUSEBUTTONDOWN and not self.done:
                    if self._is_current_human():
                        col = self._get_col(event.pos)
                        if col >= 0:
                            self._try_move(col)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self._reset()
                    elif event.key == pygame.K_q:
                        running = False

            # AI turn
            if not self.done and not self._is_current_human():
                self._ai_move()

            self._draw()
            clock.tick(30)

        pygame.quit()

    def _is_current_human(self) -> bool:
        return isinstance(self.agents[self.player], HumanAgent)

    def _get_col(self, pos) -> int:
        """Get board column from mouse position."""
        x = pos[0] - self.PADDING
        y = pos[1] - self.PADDING - 60
        if x < 0 or y < 0 or x >= self.board_w or y >= self.board_h:
            return -1
        col = x // self.CELL_SIZE
        row = y // self.CELL_SIZE
        rows, cols = self.game.state_shape
        if col < cols and row < rows:
            # For TTT, return the cell index; for C4, return column
            if isinstance(self.game, Connect4):
                return col
            return row * cols + col
        return -1

    def _try_move(self, action: int):
        valid = self.game.get_valid_actions(self.state)
        if action < len(valid) and valid[action]:
            self._apply_move(action)

    def _ai_move(self):
        agent = self.agents[self.player]
        perspective = self.state * self.player
        valid = self.game.get_valid_actions(self.state)
        action = agent.select_action(perspective, valid)
        self._apply_move(action)

    def _apply_move(self, action: int):
        self.state, self.done, self.winner = self.game.step(self.state, action, self.player)
        symbol = "X" if self.player == 1 else "O"
        self.move_history.append(f"{symbol}: {action}")

        if self.done:
            if self.winner == 0:
                self.status_msg = "Draw!"
            else:
                name = self.agents[self.winner].name
                self.status_msg = f"{name} wins!"
        else:
            self.player *= -1
            name = self.agents[self.player].name
            symbol = "X" if self.player == 1 else "O"
            self.status_msg = f"{name}'s turn ({symbol})"

    def _reset(self):
        self.state = self.game.reset()
        self.player = 1
        self.done = False
        self.winner = 0
        self.move_history = []
        self.status_msg = f"{self.agents[1].name}'s turn (X)"

    def _draw(self):
        self.screen.fill(BG_COLOUR)
        self._draw_title()
        self._draw_board()
        self._draw_info_panel()
        self._draw_status()
        pygame.display.flip()

    def _draw_title(self):
        title = self.font_title.render(self.game.name, True, TEXT_COLOUR)
        self.screen.blit(title, (self.PADDING, 15))

    def _draw_board(self):
        rows, cols = self.game.state_shape
        ox, oy = self.PADDING, self.PADDING + 60

        for r in range(rows):
            for c in range(cols):
                rect = pygame.Rect(ox + c * self.CELL_SIZE, oy + r * self.CELL_SIZE,
                                   self.CELL_SIZE, self.CELL_SIZE)
                pygame.draw.rect(self.screen, EMPTY_COLOUR, rect)
                pygame.draw.rect(self.screen, GRID_COLOUR, rect, 2)

                val = int(self.state[r, c])
                if val == 1:
                    txt = self.font_cell.render("X", True, P1_COLOUR)
                    self.screen.blit(txt, txt.get_rect(center=rect.center))
                elif val == -1:
                    txt = self.font_cell.render("O", True, P2_COLOUR)
                    self.screen.blit(txt, txt.get_rect(center=rect.center))

    def _draw_info_panel(self):
        px = self.PADDING * 2 + self.board_w
        py = self.PADDING + 60

        # Panel background
        panel_rect = pygame.Rect(px, py, self.INFO_WIDTH - self.PADDING,
                                 self.board_h)
        pygame.draw.rect(self.screen, PANEL_COLOUR, panel_rect, border_radius=10)

        # Player info
        y = py + 15
        p1_text = self.font_info.render(f"X: {self.agents[1].name}", True, P1_COLOUR)
        self.screen.blit(p1_text, (px + 15, y))
        y += 35
        p2_text = self.font_info.render(f"O: {self.agents[-1].name}", True, P2_COLOUR)
        self.screen.blit(p2_text, (px + 15, y))
        y += 45

        # Move history
        hist_title = self.font_info.render("Moves:", True, TEXT_COLOUR)
        self.screen.blit(hist_title, (px + 15, y))
        y += 30

        visible_moves = self.move_history[-8:]
        for move in visible_moves:
            colour = P1_COLOUR if move.startswith("X") else P2_COLOUR
            txt = self.font_info.render(move, True, colour)
            self.screen.blit(txt, (px + 20, y))
            y += 25

        # Controls
        y = py + self.board_h - 60
        ctrl = self.font_info.render("[R] Restart  [Q] Quit", True, HIGHLIGHT)
        self.screen.blit(ctrl, (px + 15, y))

    def _draw_status(self):
        y = self.height - 45
        colour = TEXT_COLOUR
        if self.done:
            colour = P1_COLOUR if self.winner == 1 else (P2_COLOUR if self.winner == -1 else HIGHLIGHT)
        txt = self.font_status.render(self.status_msg, True, colour)
        self.screen.blit(txt, (self.PADDING, y))


# ── CLI ───────────────────────────────────────────────────────────────────────

def _make_game(name):
    return TicTacToe() if name == "tictactoe" else Connect4()


def _make_agent(name, game, model_path=None):
    if name == "human":
        return HumanAgent()
    agents = {
        "default": lambda: DefaultAgent(game),
        "minimax": lambda: MinimaxAgent(game),
        "qlearning": lambda: QLearningAgent(game),
        "dqn": lambda: DQNAgent(game),
    }
    agent = agents[name]()
    agent.set_game(game)
    if model_path:
        agent.load(model_path)
    return agent


def main():
    parser = argparse.ArgumentParser(description="Launch game GUI")
    parser.add_argument("--game", choices=["tictactoe", "connect4"], default="tictactoe")
    parser.add_argument("--p1", default="human")
    parser.add_argument("--p2", default="default")
    parser.add_argument("--model1", default=None)
    parser.add_argument("--model2", default=None)
    args = parser.parse_args()

    game = _make_game(args.game)
    agent1 = _make_agent(args.p1, game, args.model1)
    agent2 = _make_agent(args.p2, game, args.model2)

    gui = GameGUI(game, agent1, agent2)
    gui.run()


if __name__ == "__main__":
    main()
