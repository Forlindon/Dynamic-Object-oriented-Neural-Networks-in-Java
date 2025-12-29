package net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.visuals;

import java.util.Objects;

public class Vec2d {

    double x, y;

    public Vec2d(double x, double y) {
        this.x = x;
        this.y = y;
    }

    public void addEq(Vec2d vec2d) {
        this.x += vec2d.x;
        this.y += vec2d.y;
    }

    public Vec2d add(Vec2d vec2d) {
        return new Vec2d(this.getX() + vec2d.getX(),this.getY() + vec2d.getY());
    }

    public double getX() {
        return x;
    }

    public double getY() {
        return y;
    }

    public double length() {
        return Math.sqrt(Math.pow(this.x,2) + Math.pow(this.y,2));
    }

    public void subEq(Vec2d vec) {
        this.x -= vec.x;
        this.y -= vec.y;
    }

    public Vec2d sub(Vec2d vec) {
        return new Vec2d(this.getX() - vec.getX(), this.getY() - vec.getY());
    }

    @Override
    public String toString() {
        return "Vec2d{" +
                "x=" + x +
                ", y=" + y +
                '}';
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof Vec2d vec2d)) return false;
        return getX() == vec2d.getX() && getY() == vec2d.getY();
    }

    @Override
    public int hashCode() {
        return Objects.hash(getX(), getY());
    }

    public void setX(double x) {
        this.x = x;
    }

    public void setY(double y) {
        this.y = y;
    }

    public Vec2d norm() {
        double len = this.length();
        return new Vec2d(this.getX()/len, this.getY()/len);
    }

    public Vec2d rotateBy(double degree) {
        if (this.x == 0 && this.y == 0) return new Vec2d(0,0);
        double rad = Math.toRadians(degree);
        double len = length();
        double phi = Math.atan2(getY()/len,getX()/len)+rad;
        return new Vec2d(len*Math.cos(phi),len*Math.sin(phi));
    }

    public double distance(Vec2d other) {
        return this.sub(other).length();
    }

    public double angleBetween(Vec2d other) {
        double x = this.length() * other.length();
        return Math.acos(scalar(other)/x);
    }

    public double scalar(Vec2d other) {
        return getX()*other.getX() + getY()*other.getY();
    }
}
