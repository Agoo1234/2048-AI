import numpy as np
import torch
import neat
import pygame
import random


class Game2048:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros((4, 4), dtype=int)
        self.score = 0
        self.spawn_tile()
        self.spawn_tile()
        self.game_over = False

    def spawn_tile(self):
        empty_tiles = [(i, j) for i in range(4) for j in range(4) if self.board[i, j] == 0]
        if not empty_tiles:
            return
        i, j = random.choice(empty_tiles)
        self.board[i, j] = 2 if random.random() < 0.9 else 4

    def slide_and_merge(self, row):
        new_row = [num for num in row if num != 0]
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1]:
                new_row[i] *= 2
                self.score += new_row[i]
                new_row[i + 1] = 0
        new_row = [num for num in new_row if num != 0]
        return new_row + [0] * (4 - len(new_row))

    def move(self, direction):
        if direction == 0:  # Up
            rotated = np.transpose(self.board)
        elif direction == 1:  # Down
            rotated = np.flipud(np.transpose(self.board))
        elif direction == 2:  # Left
            rotated = self.board
        elif direction == 3:  # Right
            rotated = np.fliplr(self.board)

        moved = False
        new_board = np.zeros_like(self.board)

        for i in range(4):
            row = rotated[i]
            new_row = self.slide_and_merge(row)
            new_board[i] = new_row
            if not np.array_equal(row, new_row):
                moved = True

        if direction == 0:  # Up
            self.board = np.transpose(new_board)
        elif direction == 1:  # Down
            self.board = np.transpose(np.flipud(new_board))
        elif direction == 2:  # Left
            self.board = new_board
        elif direction == 3:  # Right
            self.board = np.fliplr(new_board)

        if moved:
            self.spawn_tile()
        if not self.valid_moves_exist():
            self.game_over = True
        return moved

    def valid_moves_exist(self):
        for i in range(4):
            for j in range(4):
                if self.board[i, j] == 0:
                    return True  # Empty space exists
                if i > 0 and self.board[i, j] == self.board[i - 1, j]:
                    return True  # Mergeable vertically
                if j > 0 and self.board[i, j] == self.board[i, j - 1]:
                    return True  # Mergeable horizontally
        return False

    def get_state(self):
        return self.board.flatten()



def fitness_function(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        game = Game2048()
        genome.fitness = 0
        steps = 0

        while not game.game_over and steps < 1000:
            state = torch.tensor(game.get_state(), dtype=torch.float32).to("cuda")
            output = net.activate(state.cpu().numpy())
            move = np.argmax(output)

            previous_board = game.board.copy()
            moved = game.move(move)


            if not moved:
                genome.fitness -= 10
            elif np.array_equal(previous_board, game.board):
                genome.fitness -= 5


            genome.fitness += game.score * 0.1 + steps * 0.05
            steps += 1



def run_neat():
    config_path = "config-feedforward.txt"
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)


    winner = population.run(fitness_function, 100)  # Train for up to 100 generations
    print(f"Winner: {winner}")
    return winner, config



def visualize_play(winner, config, speed=1):
    pygame.init()
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    game = Game2048()

    screen = pygame.display.set_mode((400, 400))
    pygame.display.set_caption("2048 AI Visualization")
    font = pygame.font.Font(None, 48)
    game_over_font = pygame.font.Font(None, 72)

    clock = pygame.time.Clock()
    running = True

    while not game.game_over and running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False


        state = torch.tensor(game.get_state(), dtype=torch.float32).to("cuda")
        output = net.activate(state.cpu().numpy())
        move = np.argmax(output)


        if not game.move(move):
            print("AI Move Invalid, Selecting Random Valid Move")
            valid_moves = [i for i in range(4) if Game2048().move(i)]
            if valid_moves:
                move = random.choice(valid_moves)
                game.move(move)
                print(f"Random Valid Move Selected: {move}")
            else:
                print("No Valid Moves Left!")
        else:
            print(f"AI Move Selected: {move}")


        screen.fill((187, 173, 160))

        for i in range(4):
            for j in range(4):
                value = game.board[i, j]
                rect = pygame.Rect(j * 100, i * 100, 100, 100)
                pygame.draw.rect(screen, (205, 193, 180), rect)  # Tile background
                if value != 0:  # Only display non-zero values
                    text = font.render(str(value), True, (119, 110, 101))
                    screen.blit(text, (j * 100 + 50 - text.get_width() // 2,
                                       i * 100 + 50 - text.get_height() // 2))

        pygame.display.flip()
        clock.tick(30 * speed)


    if game.game_over:
        # screen.fill((0, 0, 0))
        text = game_over_font.render("Game Over", True, (255, 0, 0))
        screen.blit(text, (200 - text.get_width() // 2, 200 - text.get_height() // 2))
        pygame.display.flip()
        pygame.time.wait(100000)

    # pygame.quit()




if __name__ == "__main__":
    torch.set_default_tensor_type(torch.cuda.FloatTensor)  # Use GPU tensors by default
    winner, config = run_neat()  # Train the AI
    visualize_play(winner, config, speed=1)  # Visualize the final trained model
