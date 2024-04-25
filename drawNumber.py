import pygame
import numpy as np
import pygame
import numpy as np
from cnn import *
while True:
    black = (0, 0, 0)
    white = (255, 255, 255)
    WIDTH = 20
    HEIGHT = 20
    MARGIN = 0
    grid = []
    for row in range(28):
        grid.append([])
        for column in range(28):
            grid[row].append(0) 
    pygame.init()
    window_size = [560, 560]
    scr = pygame.display.set_mode(window_size)
    pygame.display.set_caption("Grid")
    done = False
    clock = pygame.time.Clock()
    turn=0
    def doit():
                pos = pygame.mouse.get_pos()
                column = pos[0] // (WIDTH + MARGIN)
                row = pos[1] // (HEIGHT + MARGIN)
                grid[row][column] = 1
                try:
                    grid[row+1][column] = 1
                except:
                    pass
                try:
                    grid[row-1][column] = 1
                except:
                    pass
                try:
                    grid[row+1][column+1] = 1
                except:
                    pass
                try:
                    grid[row-1][column-1] = 1
                except:
                    pass
                try:
                    grid[row+1][column-1] = 1
                except:
                    pass
                try:
                    grid[row][column-1] = 1
                except:
                    pass
                try:
                    grid[row][column+1] = 1
                except:
                    pass
                
    while not done:
        for event in pygame.event.get(): 
            if event.type == pygame.QUIT: 
                done = True
            elif event.type == pygame.MOUSEMOTION and turn==1:
                doit()
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                turn=1-turn
                if turn==0:
                    done=True
                    
                doit()
                #print("Click ", pos, "Grid coordinates: ", row, column)
        scr.fill(white)
        for row in range(28):
            for column in range(28):
                color = black
                if grid[row][column] == 1:
                    color = white
                pygame.draw.rect(scr,
                                 color,
                                 [(MARGIN + WIDTH) * column + MARGIN,
                                  (MARGIN + HEIGHT) * row + MARGIN,
                                  WIDTH,
                                  HEIGHT])
        clock.tick(50)
        pygame.display.flip() #update the screen
    pygame.quit()
    number = np.array([grid])
    model = ResNet18(pretrained=True, probing=False)
    model.load_state_dict(torch.load('Model.pth'))
    model.eval()
    number = torch.tensor(number).float()
    number = number.repeat(1, 3, 1, 1)  # Repeat the single channel image to have 3 channels
    prediction = model(number)
    print("prediction", prediction)
    print("prediction", torch.argmax(prediction, dim=1))
    print("prediction", torch.argmax(prediction, dim=1).item())




