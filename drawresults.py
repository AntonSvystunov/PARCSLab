from solver import Output
import pygame as pg
import sys


def draw(world_size, points, edges):
    sc = pg.display.set_mode((world_size[1] + 50, world_size[3] + 50))

    for edge in filter(lambda e: not e['deactivate'], edges):
        org = points[edge['org']]
        dest = points[edge['dest']]

        pg.draw.line(sc, (255, 255, 255), 
                    org, 
                    dest, 1)
    
    for points in points:
        pg.draw.circle(sc, (255, 0, 0), 
                    points, 
                    3)

    pg.display.update()

    while 1:
        for i in pg.event.get():
            if i.type == pg.QUIT:
                sys.exit()

    pg.time.delay(1000)

def main():
    world_size, points, edges = Output.readFrom('output (3).txt').desctruct()
    draw(world_size, points, edges)
    

if __name__ == '__main__':
    main()