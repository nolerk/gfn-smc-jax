import os
import argparse
import sys


def create_argument_combinations():
    """Create all combinations of arguments for grid search."""

    param_grid = []

    algorithms = [
        "pcula",
        # "ais",
        # "fab",
        # "smc",
        "gfn_tb",
        "gfn_tb_beta",
        # "dds",
        # "dis",
        # "pis",
        # "aft",
        # "mcd",
        # "cmcd",
        # "gbs",
        # "ula",
        # "uha",
        # "ldvi",
    ]

    targets = [
        "gmm_cube",
        # "brownian",
        # "credit",
        # "funnel",
        # "gaussian_mixture40_50d",
        # "lgcp",
        # "many_well_64d",
        # "nice_digits",
        # "nice_fashion",
        # "planar_robot_4goals",
        # "seeds",
        # "sonar",
        # "student_t_mixture_50d",
        # "gaussian_mixture40",
        # "many_well",
    ]

    target_dims = [2, 3, 5, 10, 32]
    for target_dim in target_dims:
        for target in targets:
            for algorithm in algorithms:
                param_grid.append(
                    {
                        "algorithm": algorithm,
                        "target": target,
                        "logger.comet.experiment_name": f"{target}__{target_dim if target_dim is not None else 'default'}__{algorithm}",
                    }
                )
                if target_dim is not None:
                    param_grid[-1]["target.dim"] = target_dim

    fixed_params = {}

    # Run with fixed params only
    if len(param_grid) == 0:
        param_grid = [{}]

    combinations = []

    api_key = os.getenv("COMETML_API_KEY")
    assert api_key is not None, "Comet API key is not set"
    workspace = os.getenv("COMETML_WORKSPACE")
    assert workspace is not None, "Comet workspace is not set"

    for params in param_grid:
        cmd = [
            f"COMETML_API_KEY='{api_key}' COMETML_WORKSPACE='{workspace}'",
            "python",
            "run.py",
        ]

        # Add fixed parameters
        for name, value in fixed_params.items():
            cmd.append(f"{name}={str(value)}")

        # # Add grid search parameters
        for name, value in params.items():
            cmd.append(f"{name}={str(value)}")

        # Add boolean flags (we'll create separate combinations for these)
        combinations.append(cmd)

    return [" ".join(comb) for comb in combinations]


def main(args):
    python_commands = create_argument_combinations()
    for python_command in python_commands:
        version = 8
        constraint = args.node
        name = "diffusion_samplers_bench"

        python_prefix = ""
        if args.gpus == 0:
            python_prefix = "JAX_PLATFORMS=cpu"
        python_command = f"{python_prefix} {python_command}"
        partition = ""
        if constraint in ["type_g", "type_h", "type_a"] and args.gpus > 0:
            partition = "--partition rocky"
        sbatch_command = (
            f"sbatch {partition} -A proj_1825 -G {args.gpus} --job-name={name} "
            f"-c {args.cpu_cores} "
            f"--error=sbatch_logs_{version}/{name}/%j.err --output=sbatch_logs_{version}/{name}/%j.log "
            f"--constraint='{constraint}' "
            f"--time=3:00:00 "
        )
        print(python_command, "\n\n")
        if not args.dry_run:
            os.system(f'{sbatch_command} --wrap="{python_command}"')


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--cpu_cores", type=int, default=4)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--node", type=str, default="type_a|type_b|type_c")
    args = p.parse_args()
    main(args)
