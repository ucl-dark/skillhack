import hydra
from omegaconf import DictConfig
import os
import json
import shutil
import time

from agent.polybeast import polyhydra
from os.path import dirname


@hydra.main(config_path=".", config_name="config")
def main(flags: DictConfig):
    st_home = os.environ.get("SKILL_TRANSFER_HOME")

    if st_home is None:
        print("Environment variable SKILL_TRANSFER_HOME is not set.")
        print(
            "Set this variable to the directory where "
            "trained skill networks should be stored."
        )
        print("Exiting...")
        return
    else:
        print("SKILL_TRANSFER_HOME set to", st_home)
        try:
            print(
                "Last modified skill transfer home is ",
                os.stat(st_home).st_mtime,
            )
        except BaseException:
            print("Error finding last modification to skill transfer home")

    copy_final_checkpoint = False

    # Training on a task
    if flags.model in ["foc", "ks", "hks"]:
        if flags.tasks_json is not None:
            tasks_path = "data/tasks/" + flags.tasks_json + ".json"
        else:
            tasks_path = "data/tasks/tasks.json"
        print("tasks_path", tasks_path)
        with open(
            os.path.join(dirname(dirname(dirname(__file__))), tasks_path)
        ) as tasks_json:
            tasks = json.load(tasks_json)

            if flags.env in tasks.keys():
                if flags.train_with_all_skills:
                    skills = set()
                    for task_skills in tasks.values():
                        skills.update(task_skills)
                    skills = list(skills)
                else:
                    skills = tasks[flags.env]

                skill_dirs = [
                    (st_home + "/" + skill + ".tar") for skill in skills
                ]

                for i, skill in enumerate(skills):
                    skill_dir = skill_dirs[i]

                    if not os.path.isfile(skill_dir):
                        print(
                            "Required skill",
                            skill,
                            "not found in SKILL_TRANSFER_HOME, aborting.",
                        )
                        quit(1)

                print("All skills present, begin training on task.")

                if flags.model in ["foc", "hks"]:
                    flags.foc_options_path = skill_dirs
                    flags.foc_options_config_path = [
                        st_home + "/skill_config.yaml"
                        for _ in range(len(skill_dirs))
                    ]

                elif flags.model == "ks":
                    flags.teacher_path = skill_dirs
                    flags.teacher_config_path = [
                        st_home + "/skill_config.yaml"
                        for _ in range(len(skill_dirs))
                    ]

            else:
                print("Task not found in tasks.json")
                print("Training without skills...")
    # Training a skill expert
    elif flags.model == "baseline" and "skill" in flags.env:
        assert not flags.use_lstm  # Skills should not use lstms!
        print(
            "Training skill, final checkpoint will be copied to SKILL_TRANSFER_HOME"
        )
        copy_final_checkpoint = True
    else:
        print("Model", flags.model, "does not use skills")
        print("Training without skills...")

    polyhydra.main(flags)

    if copy_final_checkpoint:
        target_path = st_home + "/" + flags.env + ".tar"
        if os.path.isfile(target_path):
            target_path = (
                st_home
                + "/"
                + flags.env
                + "_"
                + str(int(time.time()))
                + ".tar"
            )
            print(
                "Warning: skill file already exists for "
                + flags.env
                + ".  Saving checkpoint to "
                + target_path
            )
        else:
            print("Saving skill file to " + target_path)
        shutil.copyfile(flags.env + ".tar", target_path)


if __name__ == "__main__":
    main()
