import os
import pyarrow
from functools import partial

import ray
import ray.data

# from data_juicer.utils.constant import Fields
# from data_juicer.ops.base_op import Fields
# from data_juicer.core.data.ray_dataset import filter_batch

from data_juicer.ops.mapper.whitespace_normalization_mapper import WhitespaceNormalizationMapper

DEFAULT_BATCH_SIZE = 1000


def process_data(data_path, output_path):
    # ds = ray.data.read_json(data_path)

    ds = ray.data.from_items([
        {"text": "This is   a   sample text. "},
        {"text": "This is a sample text."},
        {"text": "Another    example   text. "}
    ])

    text_key = 'text'
    image_key = 'images'

    def user_custom_op(sample):
        return sample

    # =============== user custom operator ==============
    ds = ds.map(
        user_custom_op,
        runtime_env=dict(
            # conda="agent"
            conda="py310"
        )
        )

    # =============== using datajuicer operator ==============
    # add_meta_fields = mapper_op._name in TAGGING_OPS.modules

    print(ds.schema)

    # datajuicer mapper operator
    mapper_op1 = WhitespaceNormalizationMapper(text_key=text_key)
    ds = ds.map_batches(
        mapper_op1.process,
        # user_custom_op,
        batch_size=DEFAULT_BATCH_SIZE,
        num_cpus=1,
        runtime_env=dict(
            # conda="py310"
            conda='agent'
        )
        )
    # ds.write_json(output_path, force_ascii=False)
    print(f'>>>>>>>> total samples : {ds.take_all()}', flush=True)


if __name__ == '__main__':
    ray.init(address='auto')

    data_path = '/home/jiangnana.jnn/workspace/data-juicer-video-parser/data-juicer/demos/data/demo-dataset.jsonl'
    output_path = '/home/jiangnana.jnn/workspace/data-juicer-video-parser/data-juicer/outputs/2/'

    process_data(data_path, output_path)

    ds = ray.data.read_json(output_path)
    ds.filter(lambda sample: len(sample['text'])>10).show()


    # Import placement group APIs.
    from ray.util.placement_group import (
        placement_group,
        placement_group_table,
        remove_placement_group,
    )
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
    from pprint import pprint
    import time

    # Import placement group APIs.
    from ray.util.placement_group import (
        placement_group,
        placement_group_table,
        remove_placement_group,
    )
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy, NodeAffinitySchedulingStrategy

    # Initialize Ray.
    import ray

    # Create a single node Ray cluster with 2 CPUs and 2 GPUs.
    ray.init(num_cpus=2, num_gpus=2)

    # Reserve a placement group of 1 bundle that reserves 1 CPU and 1 GPU.
    pg = placement_group([{"CPU": 1, "GPU": 1}])
    # Wait until placement group is created.
    ray.get(pg.ready(), timeout=10)

    # You can also use ray.wait.
    ready, unready = ray.wait([pg.ready()], timeout=10)

    # You can look at placement group states using this API.
    print(placement_group_table(pg))

    @ray.remote(num_cpus=1)
    class Actor:
        def __init__(self):
            pass

        def ready(self):
            pass


    # Create an actor to a placement group.
    actor = Actor.options(
        num_cpus=1,
        num_gpus=1,
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg,
        )
        # scheduling_strategy=NodeAffinitySchedulingStrategy(node_is='xxx')
    ).remote()

    # Verify the actor is scheduled.
    ray.get(actor.ready.remote(), timeout=10)