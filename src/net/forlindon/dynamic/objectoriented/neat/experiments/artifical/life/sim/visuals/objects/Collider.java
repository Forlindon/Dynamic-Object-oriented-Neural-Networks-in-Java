package net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.visuals.objects;

import net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.visuals.Vec2d;

import java.lang.Object;
import java.util.Objects;

public abstract class Collider {

    Vec2d start;
    Vec2d v;

    public Collider(int x, int y, int w, int h) {
        start = new Vec2d(x,y);
        v = new Vec2d(w,h);
    }

    public int x() {
        return (int) start.getX();
    }

    public int y() {
        return (int) start.getY();
    }

    public int w() {
        return (int) v.getX();
    }

    public int h() {
        return (int) v.getY();
    }

    public boolean intersect(Collider other) {
        if (this.w() <= 0 || this.h() <= 0 || other.w() <= 0 || other.h() <= 0) return false;
        int dX = Math.abs(this.x()-other.x());
        int dY = Math.abs(this.y()-other.y());
        return (dX <= this.w() || dX <= other.w()) && (dY <= this.h() || dY <= other.h());
    }

    public boolean inRay(Vec2d start, Vec2d direction, double radius) {
        if (direction.getX() == 0 && (start.getX() < this.x() || start.getX() > this.x() + this.w())) {
            return false;
        }
        else if (direction.getY() == 0 && (start.getY() < this.y() || start.getY() > this.y() + this.h())) {
            return false;
        }
        // x
        double t1X = direction.getX() == 0 ? Double.POSITIVE_INFINITY : (double) (this.x() - start.getX()) / direction.getX();
        double t2X = direction.getX() == 0 ? Double.POSITIVE_INFINITY : (double) (this.x() + this.w() - start.getX()) / direction.getX();

        // y
        double t1Y = direction.getY() == 0 ? Double.POSITIVE_INFINITY : (double) (this.y() - start.getY()) / direction.getY();
        double t2Y = direction.getY() == 0 ? Double.POSITIVE_INFINITY : (double) (this.y() + this.h() - start.getY()) / direction.getY();

        // tMin/Max
        double tminX = Math.min(t1X, t2X);
        double tmaxX = Math.max(t1X, t2X);

        double tminY = Math.min(t1Y, t2Y);
        double tmaxY = Math.max(t1Y, t2Y);

        double tentry = Math.max(tminX, tminY);
        double texit = Math.min(tmaxX, tmaxY);

        return tentry >= 0 && tentry <= Math.min(texit,radius/direction.length());
    }

    public abstract boolean canCollide();

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof Collider collider)) return false;
        return Objects.equals(start, collider.start) && Objects.equals(v, collider.v);
    }

    @Override
    public int hashCode() {
        return Objects.hash(start, v);
    }
}
