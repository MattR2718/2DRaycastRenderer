import numpy as np
import math

import pygame
from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_ESCAPE,
    K_SPACE,
    K_RETURN,
    KEYDOWN,
    QUIT,
)

#Initialise pygame
pygame.init()

#Set constants
#MAKE SURE WIDTH, HEIGHT AND BLOCKSIZE ARE THE SAME AS IN raycasting.py
WIDTH = 800
HEIGHT = 600
BLOCKSIZE = 50

#Get clock to set fps
fps_clock = pygame.time.Clock()
FPS = 60


#Set up drawing window
screen = pygame.display.set_mode([WIDTH, HEIGHT])
pygame.display.set_caption("Raycasting")


#Sets a border around the edge of the screen
def border():
	for x in range(WIDTH//BLOCKSIZE):
		grid[x][0] = 1
		grid[x][HEIGHT//BLOCKSIZE - 1] = 1
	for y in range(HEIGHT//BLOCKSIZE):
		grid[0][y] = 1
		grid[WIDTH//BLOCKSIZE - 1][y] = 1
		
#Draws a grid on the screen based on BLOCKSIZE, WIDTH and HEIGHT
def draw_grid():
    for x in range(WIDTH // BLOCKSIZE):
        for y in range(HEIGHT // BLOCKSIZE):
            rect = pygame.Rect(x*BLOCKSIZE, y*BLOCKSIZE, BLOCKSIZE, BLOCKSIZE)
            pygame.draw.rect(screen, (0,0,0), rect, 1)


grid = np.zeros(((WIDTH//BLOCKSIZE), (HEIGHT//BLOCKSIZE)))
#print(grid)
border()
grid[1][1] = 2


running = True
place_objects = True
while running:
	# Did the user click the window close button?
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False
		#If pressed enter, stop allowing objects to be placed
		if event.type == KEYDOWN:
			if event.key == K_SPACE:
				place_objects = False
				border()
				print("SPACEBAR")
			
		#If can place objects check if mouse is pressed and update grid
		if place_objects:
			if event.type == pygame.MOUSEBUTTONDOWN:
				mouse_pos = pygame.mouse.get_pos()
				print("Mouse Pressed at: ", mouse_pos)
				if grid[mouse_pos[0]//BLOCKSIZE][mouse_pos[1]//BLOCKSIZE] == 1:
					grid[mouse_pos[0]//BLOCKSIZE][mouse_pos[1]//BLOCKSIZE] = 0
					
				elif grid[mouse_pos[0]//BLOCKSIZE][mouse_pos[1]//BLOCKSIZE] == 0:
					grid[mouse_pos[0]//BLOCKSIZE][mouse_pos[1]//BLOCKSIZE] = 1
					
	#Fill screen
	screen.fill((255,255,255))

	#Draw Grid
	draw_grid()


	#Draw objects from grid on screen
	for x in range(grid.shape[0]):
		for y in range(grid.shape[1]):
			if grid[x][y] == 1:
				pygame.draw.rect(screen, (0,0,0), (x*BLOCKSIZE, y*BLOCKSIZE, BLOCKSIZE, BLOCKSIZE))
			elif grid[x][y] == 2:
				pygame.draw.rect(screen, (255,255,255), (x*BLOCKSIZE, y*BLOCKSIZE, BLOCKSIZE, BLOCKSIZE))
	
	if not place_objects:
		np.savetxt("map.txt", grid, fmt="%0.0f")


	
	#Update display
	pygame.display.flip()
	
	fps_clock.tick(FPS)

pygame.quit()
