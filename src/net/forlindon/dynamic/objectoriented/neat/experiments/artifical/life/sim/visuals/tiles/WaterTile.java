package net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.visuals.tiles;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class WaterTile extends BasicTile {

    public static final BufferedImage WATER;

    static {
        try {
            WATER = ImageIO.read(new File("resources/water-tile.png"));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public WaterTile(int x, int y) {
        super(WATER, x, y);
    }
}
