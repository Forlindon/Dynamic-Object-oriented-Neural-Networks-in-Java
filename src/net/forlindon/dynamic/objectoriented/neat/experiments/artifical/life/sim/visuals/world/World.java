package net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.visuals.world;

import net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.neat.LadyBugSpeciesManager;
import net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.visuals.Vec2d;
import net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.visuals.objects.Bush;
import net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.visuals.objects.Entity;
import net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.visuals.objects.LadyBug;
import net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.visuals.objects.Object;
import net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.visuals.tiles.*;

import java.awt.*;
import java.util.*;
import java.util.List;

public class World {

    public final Grid TILE_MAP;
    public final List<Object> OBJECTS = new ArrayList<>();
    public final List<Object> OBJECTS_BUFFER = new ArrayList<>();
    public LadyBugSpeciesManager speciesManager;

    Random random = new Random(0);

    public World(int sizeX, int sizeY) {
        TILE_MAP = new Grid(sizeX,sizeY);
        for (int y = 0; y < sizeY; y++) {
            for (int x = 0; x < sizeX; x++) {
                int r = random.nextInt(TileType.types());
                TileType tileType = TileType.values()[r];
                TILE_MAP.add(switch (tileType) {
                    case GRASS -> new GrassTile(x,y);
                    case SAND -> new SandTile(x,y);
                    case WATER -> new WaterTile(x,y);
                });
            }
        }
        for (int i = 0; i < 4; i++) {
            WorldRule.shape(this);
        }
        WorldRule.cleanUp(this);
        this.speciesManager = new LadyBugSpeciesManager();
        addBushes(sizeX+sizeY/2);
        spawnEntities(sizeX*5);
    }

    public synchronized void draw(Graphics2D g2d) {
        this.TILE_MAP.draw(g2d);
        this.OBJECTS.forEach(object -> object.draw(g2d));
    }

    public boolean collidesWithAny(Object object) {
        return this.OBJECTS.stream().anyMatch(object1 -> object1.collidesWith(object));
    }

    int getWorldSizeX() {
        return this.TILE_MAP.cols * BasicTile.TILE_SIZE;
    }

    int getWorldSizeY() {
        return this.TILE_MAP.rows * BasicTile.TILE_SIZE;
    }

    public BasicTile get(int x, int y) {
        if (x >= cols() || y >= rows() || x < 0 || y < 0) throw new RuntimeException("Invalid Indexing: (" + x + ", " + y + ")");
        return this.TILE_MAP.get(x,y);
    }

    public void set(BasicTile basicTile) {
        this.TILE_MAP.set(basicTile);
    }

    public int rows() {
        return this.TILE_MAP.rows;
    }

    public int cols() {
        return this.TILE_MAP.cols;
    }

    public void addBushes(int n) {
        List<BasicTile> grassTiles = this.TILE_MAP.getTILE_MAP().stream().filter(basicTile -> basicTile instanceof GrassTile).toList();
        while (n > 0) {
            for (BasicTile basicTile : grassTiles) {
                if (random.nextInt(100) < 2) {
                    OBJECTS_BUFFER.add(new Bush(this, basicTile.getX()*BasicTile.TILE_SIZE + (BasicTile.TILE_SIZE - Bush.BUSH_SIZE)/2, basicTile.getY()*BasicTile.TILE_SIZE + (BasicTile.TILE_SIZE - Bush.BUSH_SIZE)/2));
                    if (--n < 1) break;
                }
            }
        }
    }

    public void spawnEntities(int n) {
        List<BasicTile> grassTiles = this.TILE_MAP.getTILE_MAP().stream().filter(basicTile -> basicTile instanceof GrassTile).toList();
        while (n > 0) {
            for (BasicTile basicTile : grassTiles) {
                if (random.nextInt(100) < 2) {
                    OBJECTS_BUFFER.add(new LadyBug(this, basicTile.getX()*BasicTile.TILE_SIZE + (BasicTile.TILE_SIZE - LadyBug.LADY_BUG_SIZE)/2, basicTile.getY()*BasicTile.TILE_SIZE + (BasicTile.TILE_SIZE - LadyBug.LADY_BUG_SIZE)/2));
                    if (--n < 1) break;
                }
            }
        }
    }

    public boolean isInBounds(Vec2d start, Vec2d d) {
        return this.getWorldSizeX() > start.getX() + d.getX() &&
                this.getWorldSizeY() > start.getY() + d.getY() &&
                start.getX() >= 0 &&
                start.getY() >= 0;
    }

    public void tickEverything() {
        synchronized (OBJECTS) {
            this.OBJECTS.forEach(Object::tick);
        }
        this.spawnObjects();
        this.kill();
        this.speciesManager.sort();
    }

    public synchronized void kill() {
        this.OBJECTS.removeIf(object -> object instanceof Entity entity && entity.isDead());
    }

    public synchronized  <T extends Object> List<Object> getObjects(Class<T> objectClass) {
        return this.OBJECTS.stream().filter(object -> object.getClass() == objectClass).toList();
    }

    public synchronized void spawnObjects() {
        this.OBJECTS.addAll(this.OBJECTS_BUFFER);
        this.OBJECTS_BUFFER.clear();
    }

}
