package net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.visuals.objects;

import net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.visuals.Vec2d;
import net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.visuals.tiles.BasicTile;
import net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.visuals.world.World;

import java.awt.*;
import java.awt.geom.AffineTransform;
import java.awt.image.BufferedImage;

public abstract class Object extends Collider implements Comparable<Object> {

    World world;

    public Object(World world, int x, int y, int w, int h) {
        super(x, y, w, h);
        this.world = world;
    }

    public boolean collidesWith(Object other) {
        return this.canCollide() && other.canCollide() && this.intersect(other) && this != other;
    }

    public abstract void draw(Graphics2D g2d);

    public boolean collidesWithAny() {
        return this.world.collidesWithAny(this);
    }

    public void move(Vec2d vec) {
        if (canMove(vec)) {
            this.start.addEq(vec);
        }
    }

    public boolean canMove(Vec2d vec2d) {
        return !collidesWithAny() && world.isInBounds(this.start.add(vec2d),this.v);
    }

    @Override
    public int compareTo(Object o) {
        return Integer.compare(this.y()+this.h(),o.y()+o.h());
    }

    public void tick() {}

    public static void draw(BufferedImage bufferedImage, Graphics2D g, int x, int y, int w, int h, Vec2d vec2d) {
        Graphics2D g2d = (Graphics2D) g.create();

        int centerX = x + w / 2;
        int centerY = y + h / 2;


        double angle = Math.atan2(vec2d.getY(), vec2d.getX()) + Math.PI/2;; // Sprite is rotated by 90° by default

        if (!Double.isNaN(angle) && vec2d.length() != 0) g2d.rotate(angle, centerX, centerY);
        g2d.drawImage(bufferedImage, x, y, w, h, null);

        g2d.dispose();
    }

    public BasicTile getTile() {
        int x = (x()+w()/2) / BasicTile.TILE_SIZE;
        int y = (y()+h()/2) / BasicTile.TILE_SIZE;
        return this.world.get(x,y);
    }

    public Vec2d getCenter() {
        return this.start.add(new Vec2d(w()/2.0,h()/2.0));
    }

    public double encode() {
        return 0;
    }
}
