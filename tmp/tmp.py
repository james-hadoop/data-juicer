"""DataJuicer分布式框架使用的Ray框架，所以算子天然支持Ray分布式。
这里展示如何在Ray框架上使用DataJuicer算子进行数据处理。
"""

import os
import pyarrow
from loguru import logger
from functools import partial

import ray
import ray.data

from data_juicer.utils.constant import Fields
from data_juicer.ops.base_op import DEFAULT_BATCH_SIZE, Fields
from data_juicer.core.data.ray_dataset import filter_batch
from data_juicer.ops.filter.text_length_filter import TextLengthFilter
from data_juicer.ops.mapper.whitespace_normalization_mapper import WhitespaceNormalizationMapper

DEFAULT_BATCH_SIZE = 1000


from ray.data._internal.util import get_compute_strategy
v = get_compute_strategy(lambda n: n, concurrency=1)
import pickle
pickle.dumps(v)
exit(0)

def prepare_ds_for_dj_op(ds, add_meta_fileds=True, add_stats_fields=True):
    columns = ds.columns()

    # if op._name in TAGGING_OPS.modules and Fields.meta not in ds.columns():
    if add_meta_fileds and Fields.meta not in ds.columns():

        def process_batch_arrow(table: pyarrow.Table):
            new_column_data = [{} for _ in range(len(table))]
            new_table = table.append_column(Fields.meta, [new_column_data])
            return new_table

        ds = ds.map_batches(
            process_batch_arrow, batch_format="pyarrow", batch_size=DEFAULT_BATCH_SIZE
        )
    
    if add_stats_fields and Fields.stats not in columns:

        def process_batch_arrow(table: pyarrow.Table):
            new_column_data = [{} for _ in range(len(table))]
            new_talbe = table.append_column(Fields.stats, [new_column_data])
            return new_talbe

        ds = ds.map_batches(
            process_batch_arrow, batch_format="pyarrow", batch_size=DEFAULT_BATCH_SIZE
        )

    return ds


if __name__ == '__main__':
    data_path = 'demos/data/demo-dataset.jsonl'
    output_path = './tmp/outputs'

    ray.init(address='auto')

    # ds = ray.data.read_csv(data_path)
    ds = ray.data.from_items(
        [
            {"text": "This paper proposed a novel method on LLM pretraining."},
            {"text": "欢迎来到阿里巴巴！"},
            {"text": "Today is Sunday and it's a happy day!"}
        ] * 10
    )

    text_key = 'text'
    image_key = 'images'

    def user_custom_op(sample):
        return sample

    # =============== user custom operator ==============
    ds = ds.map(user_custom_op)

    # =============== using datajuicer operator ==============
    # add_meta_fileds = mapper_op._name in TAGGING_OPS.modules
    ds = prepare_ds_for_dj_op(ds, add_meta_fileds=False, add_stats_fields=True)

    # datajuicer mapper operator
    mapper_op1 = WhitespaceNormalizationMapper(text_key=text_key)
    if mapper_op1.is_batched_op():
        ds = ds.map_batches(
            mapper_op1.process,
            batch_size=1000,
            num_cpus=1)
    else:
        ds = ds.map(mapper_op1.process, num_cpus=1)

    # datajuicer filter operator
    filter_op1 = TextLengthFilter(min_len=1, text_key=text_key)
    if filter_op1.is_batched_op():
        ds = ds.map_batches(
            filter_op1.compute_stats,
            batch_size=1000,
            num_cpus=1)

        ds = ds.map_batches(
            partial(filter_batch, filter_func=filter_op1.process),
            batch_format="pyarrow",
            zero_copy_batch=True,
            batch_size=DEFAULT_BATCH_SIZE,
        )
    else:
        ds = ds.map(filter_op1.compute_stats, num_cpus=1)
        ds = ds.filter(filter_op1.process, num_cpus=1)

    ds.write_json(output_path, force_ascii=False)
    print(f'>>>>>>>> total samples : {ds.count()}', flush=True)
