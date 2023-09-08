# Adapt
**A**ctuator  **D**egeneration **A**da**p**tation **T**ransformer. In submission progress of DAI 2023.

## Environment Require

-   Nvidia IsaacGym
-   PyTorch

Details could be viewd in `requirement.txt`.

## Teacher Policy

-   To train a teacher policy, please follow the example command:

    ```bash
    python legged_gym/scripts/train.py --task a1_amp --sim_device $DEVICE --rl_device $DEVICE \
    	--experiment_name $EXP --rum_name $RUN --max_iteration $ITER --joint $JOINT --seed $SEED
    ```

​		`$JOINT` means the id of joint whose actuator is suffering the degeneration.

-   To evaluate, please follow the example command:

    ```bash
    python legged_gym/scripts/evaluate.py --task a1_amp --sim_device $DEVICE --rl_device $DEVICE \
    	--experiment_name $EXP --load_run $RUN --checkpoint $CHECKPOINT --file_name $FILE --joint $JOINT
    ```

    If you want to test the performance over all 12 situations, please set `$JOINT=-1`.

-   To collect dataset, please follow the example command:

    ```bash
    python legged_gym/scripts/collect.py --task a1_amp --sim_device $DEVICE --rl_device $DEVICE \
    	--experiment_name $EXP --load_run $RUN --checkpoint $CHECKPOINT --file_name $FILE --joint $JOINT
    ```

-    To visualize the performance, please follow the example command:

    ```   bash
    python legged_gym/scripts/play.py --task a1_amp --sim_device $DEVICE --rl_device $DEVICE \
    	--experiment_name $EXP --load_run $RUN --checkpoint $CHECKPOINT --joint $JOINT --rate $RATE
    ```

    `$RATE` means the degeneration rate, `-1` for randomization.


## Student Policy

-   To train a student policy, please follow the example command:

    ```bash
    python scripts/train_Adapt.py
    ```

    The detail args could be viewed in `args.yaml`.

-   To evaluate, please follow the example command:

    ```bash
    python scripts/evaluate_Adapt.py
    ```

    The detail args could be viewed in `test_args.yaml`.

-   To visualize the performance, please follow the example command:

    ```bash
    python scripts/play_Adapt.py
    ```

    The detail args could be viewed in `test_args.yaml`.
