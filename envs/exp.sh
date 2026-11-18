# Base experiment definition shared by every controller of the paper. Trimmed from the exp.sh of
# legged_env (MIT, see LICENSE) to the functions that envs/exp_onepass.sh uses.

dukehumanoid_diffusion(){
    base
    change_hydra_dir

    PLAY_ARGS+=(
        headless=false

        num_envs=2
        task.env.dataPublisher.enable=true

        task.env.learn.episodeLength_s=90

        task.env.terrain.maxInitMapLevel=1
        task.env.terrain.numLevels=3
        task.env.terrain.numTerrains=3
        task.env.terrain.terrainType=plane
        task.env.terrain.terrainProportions=[1,1,2,1,1,1,1,2,0]

        ++task.env.renderFPS=50

        task.env.randomize.baseInertiaOrigin.enable=false
        task.env.randomize.push.enable=false
        task.env.randomize.friction.enable=false
        task.env.randomize.initDofPos.enable=false
        task.env.randomize.initDofVel.enable=false
        task.env.randomize.erfi.enable=false
        task.env.randomize.dof_strength.enable=false
        task.env.randomize.default_dof_pos.enable=false
        task.env.randomize.link_mass.enable=false
        task.env.randomize.link_inertia.enable=false
        task.env.randomize.body_force.enable=false
        task.env.randomize.baseMass.enable=false
        task.env.learn.addNoise=false
    )

    REWARD_ARGS=(
        task=Biped

        task.env.keep_still_at_zero_command=false
        ++task.env.max_observation_delay_steps=1
        task.env.learn.guided_contact.phase_freq=1.25

        train.params.config.horizon_length=32
        train.params.config.mini_epochs=8
        train.params.config.max_epochs=5000

        # Use diffusion-based policy
        train=BipedPPODiffusion
        task.env.asymmetric_observations=True
        task.env.stateNames=[linearVelocity,angularVelocity,projectedGravity,commands,dofPosition,dofVelocity,actions,contactTarget,contact,phase]
        task.env.observationNames=[angularVelocity,projected_gravity_filtered,commands,dofPosition,dofVelocity,actions,contactTarget,phase]
        ++task.env.num_stacked_obs_frame=5
        ++task.env.num_stacked_state_frame=3

        task.env.randomCommandVelocityRanges.linear_x=[0,1]
        task.env.randomCommandVelocityRanges.linear_y=[0,0]
        task.env.randomCommandVelocityRanges.yaw=[0,0]

        task.env.randomize.baseInertiaOrigin.enable=true
        task.env.randomize.baseInertiaOrigin.range=[[-0.02,0.02],[-0.02,0.02],[-0.02,0.02]]
        task.env.randomize.link_inertia.enable=true
        task.env.randomize.link_mass.enable=true

        task.env.randomize.push.enable=true
        task.env.randomize.push.interval_s=8

        task.env.randomize.friction.range=[0.2,1.2]
        task.env.randomize.baseMass.enable=true
        task.env.randomize.baseMass.range=[-0.5,5.0]
        task.env.randomize.dof_strength.enable=true
        task.env.randomize.dof_strength.range=[0.95,1.02]

        task.env.randomize.action_delay.enable=true

        task.env.learn.allowKneeContacts=false

        task.env.learn.reward.lin_vel.exp_scale=-4.0
        task.env.learn.reward.lin_vel.normalize_by=[1,1,0.1]
        task.env.learn.reward.ang_vel.exp_scale=-8.0

        task.env.learn.reward.orientation.scale=-20

        task.env.learn.reward.dof_force_target.scale=0.05
        task.env.learn.reward.dof_force_target.exp_scale=-1
        task.env.learn.reward.dof_force_target_swing.scale=0
        task.env.learn.reward.dof_force_target_swing.exp_scale=-1.0

        task.env.learn.reward.dof_acc.scale=0.1
        task.env.learn.reward.dof_acc.exp_scale=-1e-4

        task.env.learn.reward.dof_vel.scale=0
        task.env.learn.reward.dof_pos.scale=-0.05
        task.env.learn.reward.dof_pow.scale=0

        task.env.learn.reward.base_height.scale=0.1
        task.env.learn.reward.base_height.exp_scale=-2000
        task.env.baseHeightTargetOffset=0

        task.env.learn.reward.air_time.scale=1
        task.env.learn.reward.air_time.offset=-0.3
        task.env.learn.reward.stance_time.scale=0
        task.env.learn.reward.single_contact.scale=0
        task.env.learn.reward.contact_force.scale=0
        task.env.learn.reward.slip.scale=0
        task.env.learn.reward.impact.scale=0
        task.env.learn.reward.stumble.scale=0
        task.env.learn.reward.collision.scale=0
        task.env.learn.reward.action.scale=0

        task.env.learn.reward.foot_pos.scale=0.1
        task.env.learn.reward.foot_pos.exp_scale=-1000
        task.env.learn.reward.foot_pos.normalize_by=[1,1,0]

        task.env.learn.reward.foot_forward.scale=0.1
        task.env.learn.reward.foot_forward.exp_scale=-10

        task.env.learn.reward.foot_orientation.scale=0.1
        task.env.learn.reward.foot_orientation.exp_scale=-8

        task.env.learn.reward.foot_height.scale=1.0
        task.env.learn.reward.foot_height.clamp_max=0.05

        task.env.learn.reward.action_rate.scale=0
        task.env.learn.reward.action_rate.exp_scale=-0.001

        task.env.learn.reward.dof_limit.scale=-100.0
        task.env.learn.reward.dof_limit.margin=0.1

        task.env.learn.reward.should_contact.scale=0.5

        task.env.terrain.difficultySale=0.2
        task.env.terrain.curriculum=false
        task.env.terrain.slopeTreshold=0.0
        task.env.terrain.terrainProportions=[1,1,2,1,1,1,1,2,0]
        task.env.terrain.terrainType=plane

        task.env.learn.dofPositionNoise=0.03
        task.env.learn.foot_contact_threshold=20
        task.env.learn.guided_contact.phase_start_with_swing=true

        ++task.env.urdfAsset.AssetOptions.vhacd_enabled=true
        ++task.env.urdfAsset.AssetOptions.vhacd_params.max_convex_hulls=3
        ++task.env.urdfAsset.AssetOptions.vhacd_params.max_num_vertices_per_ch=16
    )

    ROBOT_SPECIFIC_ARGS+=(
        ++task.env.assetDofProperties.velocity=10
        task.env.control.stiffness=[80,80,80,80,60,80,80,80,80,60]
        task.env.control.damping=[8,8,8,8,5,8,8,8,8,5]
        ++task.env.assetDofProperties.damping=0
        ++task.env.assetDofProperties.armature=[0.2100,0.4200,0.3800,0.2100,0.0750,0.2100,0.4200,0.3800,0.2100,0.0750]
        ++task.env.assetDofProperties.friction=[0.01,0.01,0.01,0.04,0.04,0.01,0.01,0.01,0.04,0.04]
        ++task.env.defaultJointPositions=[0.000,0.175,0.100,0.387,-0.213,0.000,-0.175,-0.100,-0.387,0.213]
    )
    BASE_ARGS+=(${REWARD_ARGS[@]})
    BASE_ARGS+=(${ROBOT_SPECIFIC_ARGS[@]})

    # Set checkpoint only if called directly, not as parent of a child function
    if [[ "${FUNCNAME[1]}" == "main" || "${FUNCNAME[1]}" == "source" || -z "${FUNCNAME[1]}" ]]; then
        PLAY_ARGS+=(
            checkpoint=$(get_latest_checkpoint dukehumanoid_diffusion)
        )
    fi
}

base(){
    checkpoint=null
    ENTRY_POINT=train.py
    PLAY_ARGS=(
        test=play
    )
    
    TRAIN_ARGS=(
        headless=True
    )
    EXPORT_ARGS=(
        test=export
        num_envs=1
    )
    BASE_ARGS=()
    KEYBOARD_ARGS=(
        task.env.viewer.keyboardOperator=true
    )
}

change_hydra_dir(){
    BASE_ARGS+=(
        "hydra.run.dir=..//outputs//\$\{task_name\}//\$\{test\}//${FUNCNAME[1]}"
    )
}

get_latest_checkpoint(){
    local exp_name="${1:-${FUNCNAME[1]}}"
    # Get the directory where this script is located
    local script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    local base_dir="${script_dir}/../outputs/Biped/train/${exp_name}/runs"
    local latest_checkpoint=""
    
    if [ -d "$base_dir" ]; then
        # Find the most recently modified .pth file
        latest_checkpoint=$(find "$base_dir" -name "*.pth" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    fi
    
    if [ -n "$latest_checkpoint" ]; then
        # Quote the path to handle spaces for Hydra
        echo "'${latest_checkpoint}'"
    else
        echo "null"
    fi
}
