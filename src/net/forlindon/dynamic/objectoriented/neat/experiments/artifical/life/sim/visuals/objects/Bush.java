package net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.visuals.objects;

import net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.visuals.Vec2d;
import net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.visuals.tiles.BasicTile;
import net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.visuals.world.World;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class Bush extends Object {

    public static final int BUSH_SIZE = (int)(BasicTile.TILE_SIZE * 0.8125);
    public static final BufferedImage BUSH_FULL, BUSH_2, BUSH_1, BUSH_0;
    static {
        try {
            BUSH_FULL = ImageIO.read(new File("resources/bush.png"));
            BUSH_2 = ImageIO.read(new File("resources/bush-2.png"));
            BUSH_1= ImageIO.read(new File("resources/bush-1.png"));
            BUSH_0 = ImageIO.read(new File("resources/bush-0.png"));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    int capacity = 0;
    int tick = 0;

    private static final int TIME_TO_GROW = 50;
    public static final int HUNGER_PER_BERRY = 55;

    public Bush(World world, int x, int y) {
        super(world, x, y, BUSH_SIZE, BUSH_SIZE);
    }

    public void draw(Graphics2D g2d) {
        g2d.drawImage(capacity == 3 ? BUSH_FULL : capacity == 2 ? BUSH_2 : capacity == 1 ? BUSH_1 : BUSH_0, x(), y(), BUSH_SIZE, BUSH_SIZE, null);
    }

    @Override
    public boolean canCollide() {
        return false;
    }

    @Override
    public void move(Vec2d vec) {
    }

    @Override
    public void tick() {
        super.tick();
        this.tick++;
        this.tick %= TIME_TO_GROW;
        if (capacity < 3 && tick == 0) {
            capacity++;
        }
    }

    public void eatBerry(Entity e) {
        if (this.capacity > 0) {
            this.capacity--;
            e.hunger = Math.min(e.maxHunger, e.hunger + HUNGER_PER_BERRY);
        }
    }

    @Override
    public double encode() {
        return 1;
    }
}
