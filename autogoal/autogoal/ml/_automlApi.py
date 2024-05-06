from autogoal.ml import AutoML
from typing import List, Tuple
from autogoal.kb import Pipeline, SemanticType, build_pipeline_graph
from pathlib import Path
import shutil

from autogoal.utils import (
    generate_production_dockerfile,
    nice_repr,
    create_zip_file,
    ensure_directory,
)


class AutoMLApi(AutoML):


    def __init__(self,
        input=None,
        output=None,
        random_state=None,
        search_algorithm=None,
        search_iterations=100,
        include_filter=".*",
        exclude_filter=None,
        validation_split=0.3,
        errors="warn",
        cross_validation="median",
        cross_validation_steps=3,
        registry=None,
        objectives=None,
        remote_sources: List[Tuple[str, int] or str] = None,
        **search_kwargs,
    ):

        super().__init__(
            input=input,
            output=output,
            random_state=random_state,
            search_algorithm=search_algorithm,
            search_iterations=search_iterations,
            include_filter=include_filter,
            exclude_filter=exclude_filter,
            validation_split=validation_split,
            errors=errors,
            cross_validation=cross_validation,
            cross_validation_steps=cross_validation_steps,
            registry=registry,
            objectives=objectives,
            remote_sources=remote_sources,
            **search_kwargs,
        )
    
    def export_portable(
        self, path=None, pipelines: List[Pipeline] = None, generate_zip=False, identifier=None, filename=None
    ):
        """
        Generates a portable set of files that can be used to export the model into a new Docker image.

        :param path: Optional. The path where the generated portable set of files will be saved. If not specified, the files will be saved to the current working directory.
        :param generate_zip: Optional. A boolean value that determines whether a zip file should be generated with the exported assets. If True, a zip file will be generated and its path will be returned.
        :return: If generate_zip is False, the path to the assets directory. If generate_zip is True, the path of the generated zip file containing the exported assets.
        """
        if path is None:
            path = os.getcwd()
        
        datapath = ""
        if identifier is not None:
            datapath = f"{path}/{identifier}"
        else:
            datapath = f"{path}/autogoal-export"

        final_path = Path(datapath)
        if final_path.exists():
            shutil.rmtree(datapath)

        self.folder_save(final_path, pipelines)

        makefile = open(final_path / "makefile", "w")
        makefile.write(
            """
build:
	docker build --file ./dockerfile -t autogoal:production .
	docker save -o autogoal-prod.tar autogoal:production

serve: build
	docker run -p 8000:8000 autogoal:production

        """
        )
        makefile.close()

        if generate_zip:
            if filename is None:
                filename = create_zip_file(datapath, "production_assets")
                datapath = f"{path}/{filename}.zip"
            else:
                filename = create_zip_file(datapath, filename)
                datapath = f"{path}/{filename}.zip"

        print("generated assets for production deployment")

        return datapath

    

