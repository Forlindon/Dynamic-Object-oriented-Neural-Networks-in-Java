package net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.visuals.tiles;

import java.awt.*;
import java.awt.image.BufferedImage;

public abstract class BasicTile {

    public static final int TILE_SIZE = 20;

    BufferedImage bufferedImage;

    int x, y;

    public BasicTile(BufferedImage bufferedImage, int x, int y) {
        this.bufferedImage = bufferedImage;
        this.x = x;
        this.y = y;
    }

    public void draw(Graphics2D graphics2D) {
        graphics2D.drawImage(this.bufferedImage, this.x*TILE_SIZE, this.y*TILE_SIZE, TILE_SIZE, TILE_SIZE, null);
    }

    public int getX() {
        return x;
    }

    public int getY() {
        return y;
    }
}
