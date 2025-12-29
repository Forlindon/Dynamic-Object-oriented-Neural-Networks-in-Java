package net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.visuals.tiles;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class GrassTile extends BasicTile {

    public static final BufferedImage BASIC_GRASS;
    public static final BufferedImage GRASS_FLOWER;
    public static final BufferedImage GRASS_MORE_FLOWERS;

    static {
        try {
            BASIC_GRASS = ImageIO.read(new File("resources/basic-grass-tile.png"));
            GRASS_FLOWER = ImageIO.read(new File("resources/grass-tile-flower.png"));
            GRASS_MORE_FLOWERS = ImageIO.read(new File("resources/grass-tile-more-flowers.png"));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public GrassTile(int x, int y) {
        super(Math.random() < 0.5 ? BASIC_GRASS : Math.random() < 0.5 ? GRASS_FLOWER : GRASS_MORE_FLOWERS, x, y);
    }


}
