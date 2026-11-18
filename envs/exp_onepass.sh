# Training runs of the paper: ONE policy trained on a mixed-terrain curriculum.
#
# Usage, from envs/:
#   VARIANT=full SEED=1 ./run.sh onepass          # train
#   VARIANT=full SEED=1 ./run.sh onepass -r       # dry run: print the command only
#   VARIANT=pilot SEED=1 ./run.sh onepass         # 500-iteration check run
# Checkpoints: ../outputs/Biped/train/onepass_${VARIANT}_seed${SEED}/runs/onepass_${VARIANT}_seed${SEED}/nn/
#
# Everything shared by all variants (robot, PD gains, rewards, observations, domain randomisation,
# PPO horizon/epochs) is inherited unchanged from dukehumanoid_diffusion() in exp.sh.

onepass(){
    dukehumanoid_diffusion
    local v=${VARIANT:-full}
    local s=${SEED:-1}
    local name=onepass_${v}${RUN_TAG:+_${RUN_TAG}}_seed${s}   # RUN_TAG: optional suffix for tuning runs
    local epochs=${MAX_EPOCHS:-5000}
    [[ "$v" == "pilot" ]] && epochs=${MAX_EPOCHS:-500}

    BASE_ARGS+=(
        "hydra.run.dir=..//outputs//Biped//train//${name}"
        seed=${s}
        experiment=${name}
        ++full_experiment_name=${name}
        train=BipedPPOOnePass
        train.params.config.max_epochs=${epochs}

        # unified mixed-terrain curriculum (terrain.py kind order:
        # rough_up, rough_down, rough_flat, stair_up, stair_down, smooth_up, smooth_down, discrete, stepping_stone)
        task.env.terrain.terrainType=trimesh
        task.env.terrain.curriculum=true
        task.env.terrain.numLevels=10
        task.env.terrain.numTerrains=20
        task.env.terrain.maxInitMapLevel=0
        task.env.terrain.difficultySale=1.0
        task.env.terrain.slopeTreshold=0.5
        task.env.terrain.terrainProportions=[1,1,2,2,2,1,1,2,0]
        task.env.terrain.slope=0.4
        task.env.terrain.stair.height=0.15
        task.env.terrain.uniform.height=0.15
        task.env.terrain.discrete.height=0.15
        ++task.env.terrain.curriculumPerEnvDistance=true   # curriculum by each robot's own commanded distance
        ++task.env.terrain.logLevelPerKind=true            # TensorBoard Episode/terrain_level_<kind> (Fig. 4b)
    )

    case "$v" in
        full|pilot)            ;;
        ppo_only)              BASE_ARGS+=(train.params.config.onepass.enable_dn=false) ;;
        ppo_mlp)               BASE_ARGS+=(train=BipedPPOMLPMatched) ;;
        ppo_mlp_joint)         BASE_ARGS+=(train=BipedPPOMLPMatched train.params.algo.name=a2c_continuous_joint_mlp ++train.params.config.onepass.lambda_joint=0.1) ;;
        abl_self_distill)      BASE_ARGS+=(train.params.config.onepass.dn_target=self_distill) ;;
        abl_no_enc_grad)       BASE_ARGS+=(train.params.config.onepass.dn_detach_encoder=true) ;;
        abl_no_terrain_latent) BASE_ARGS+=(train.params.network.diffusion.zero_terrain_latent=true) ;;
        abl_no_stair_data)     BASE_ARGS+=(task.env.terrain.terrainProportions=[1,1,2,0,0,1,1,2,0]) ;;
        abl_linear_schedule)   BASE_ARGS+=(train.params.network.diffusion.schedule=linear) ;;
        abl_no_resblocks)      BASE_ARGS+=(train.params.network.diffusion.no_resblocks=true) ;;
        abl_no_joint_loss)     BASE_ARGS+=(train.params.config.onepass.enable_joint=false) ;;
        *) echo "onepass: unknown VARIANT=$v" >&2; return 1 ;;
    esac
    BASE_ARGS+=(${EXTRA_OVERRIDES})
}
