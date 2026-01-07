package net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.visuals.objects;

import net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.visuals.Vec2d;
import net.forlindon.dynamic.objectoriented.neat.experiments.artifical.life.sim.visuals.world.World;

public abstract class Entity extends Object {

    Vec2d velocity = new Vec2d(0,0);

    double health;
    double maxHealth;
    double hunger;
    double hungerThreshold;
    double maxHunger;
    double reproductionThreshold;
    double reproductionTickThreshold;
    int maxAge;
    int age;
    boolean dead = false;

    public static final double PASSIVE_HUNGER = 0.5;

    public Entity(World world, int x, int y, int w, int h) {
        super(world, x, y, w, h);
    }

    @Override
    public boolean canCollide() {
        return true;
    }

    @Override
    public void move(Vec2d vec) {
        if (canMove(vec)) {
            velocity.setX(vec.getX());
            velocity.setY(vec.getY());
        }
        else {
            velocity.setX(vec.getX()*1e-10);
            velocity.setY(vec.getY()*1e-10);
        }
        super.move(this.velocity);
    }

    public boolean isDead() {
        return this.dead;
    }

    public void kill() {
        this.dead = true;
    }

    @Override
    public void tick() {
        super.tick();
        if (isDead()) return;
        if (health <= 0 || age++ >= maxAge) {
            kill();
            return;
        }
        this.hunger+=PASSIVE_HUNGER;
    }
}
