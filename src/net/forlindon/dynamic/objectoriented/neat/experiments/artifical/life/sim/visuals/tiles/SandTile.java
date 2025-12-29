package net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.visuals.tiles;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class SandTile extends BasicTile {

    public static final BufferedImage SAND;

    static {
        try {
            SAND = ImageIO.read(new File("resources/sand-tile.png"));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public SandTile(int x, int y) {
        super(SAND, x, y);
    }
}
