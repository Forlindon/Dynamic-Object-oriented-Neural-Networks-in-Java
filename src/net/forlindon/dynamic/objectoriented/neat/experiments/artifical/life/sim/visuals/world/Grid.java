package net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.visuals.world;

import net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.visuals.tiles.BasicTile;

import java.awt.*;
import java.util.ArrayList;
import java.util.List;

public class Grid {

    int rows, cols;

    private final List<BasicTile> TILE_MAP;

    public Grid(int sizeX, int sizeY) {
        this(sizeY,sizeX,new ArrayList<>(sizeX*sizeY));
    }

    private Grid(int r, int c, List<BasicTile> tiles) {
        this.rows = r;
        this.cols = c;
        this.TILE_MAP = tiles;
    }

    public void set(BasicTile bt) {
        this.TILE_MAP.set(bt.getY()*cols+bt.getX(),bt);
    }

    public BasicTile get(int x, int y) {
        if (x >= cols || y >= rows || x < 0 || y < 0) throw new RuntimeException("Invalid Indexing: (" + x + ", " + y + ")");
        return this.TILE_MAP.get(y*cols+x);
    }

    public Grid copy() {
        return new Grid(this.rows,this.cols,new ArrayList<>(this.TILE_MAP));
    }

    public void draw(Graphics2D g2d) {
        this.TILE_MAP.forEach(basicTile -> basicTile.draw(g2d));
    }


    public void add(BasicTile basicTile) {
        this.TILE_MAP.add(basicTile);
    }

    public void fill(Grid g) {
        if (rows != g.rows || cols != g.cols) throw new RuntimeException("Shape mismatch of grids");
        for (int i = 0; i < TILE_MAP.size(); i++) {
            this.TILE_MAP.set(i, g.TILE_MAP.get(i));
        }
    }

    public List<BasicTile> getTILE_MAP() {
        return TILE_MAP;
    }
}
