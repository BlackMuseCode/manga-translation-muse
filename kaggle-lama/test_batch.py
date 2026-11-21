"""Offline contract tests; no Kaggle credentials or GPU are required."""
import base64
import enum
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
import zipfile

import batch_runner as worker
import client


def png(color='white', size=(12, 18), mode='RGB'):
    with worker.Image.new(mode, size, color) as image:
        stream = io.BytesIO()
        image.save(stream, 'PNG')
        return stream.getvalue()


class FakeEngine:
    def process(self, image, **options):
        return dict(width=image.width, height=image.height, boxes=[],
            inpainted_image_b64=base64.b64encode(png(size=image.size)).decode(),
            mask_b64=base64.b64encode(png(0, image.size, 'L')).decode(),
            precise_mask_b64=base64.b64encode(png(17, image.size, 'L')).decode())


class ContractTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parent/'.work'
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.root = Path(self.temp.name)
        self.images = self.root/'images'
        self.images.mkdir()
        (self.images/'中文页.png').write_bytes(png())

    def tearDown(self):
        self.temp.cleanup()

    def prepare(self, options=None):
        self.folder = self.root/'job'
        self.state = client.prepare(self.images, self.folder, options,
                                   job_id='lama-test-12345678')
        return self.state

    def run_job(self, engine=None, **kwargs):
        self.prepare()
        self.state['runtime_sha256'] = 'a'*64
        with zipfile.ZipFile(self.folder/'task.bin') as archive:
            request = worker.read_task(archive, self.state['job_id'], self.state['request_sha256'])
            result = worker.run_batch(archive, request, engine or FakeEngine(), self.root/'outputs',
                provenance={'runtime_sha256': 'a'*64}, **kwargs)
        return result

    def test_default_parameters_match_api(self):
        self.assertEqual(worker.validate_options({}), dict(mask_dilate_size=10,
            mask_box_expand_ratio=20., extra_boxes=()))

    def test_extra_box_clipping(self):
        options = worker.validate_options({'extra_boxes': [[-8, 2, 500, 19]]}, (12, 18))
        self.assertEqual(options['extra_boxes'], ((0, 2, 12, 18),))

    def test_bad_parameters_rejected(self):
        for values in ({'foo': 1}, {'mask_dilate_size': True}, {'mask_dilate_size': 65},
            {'mask_box_expand_ratio': float('nan')}, {'mask_box_expand_ratio': 101},
            {'extra_boxes': [[1., 2, 3, 4]]}, {'extra_boxes': [[4, 2, 1, 5]]},
            {'extra_boxes': [[0, 0, 1, 2]]*65}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                worker.validate_options(values)

    def test_box_empty_after_clipping(self):
        with self.assertRaises(ValueError):
            worker.validate_options({'extra_boxes': [[20, 2, 30, 4]]}, (12, 18))

    def test_prepare_preserves_bytes_and_names(self):
        state = self.prepare({'pages': {'中文页.png': {'mask_dilate_size': 7}}})
        with zipfile.ZipFile(self.folder/'task.bin') as archive:
            request = worker.read_task(archive, state['job_id'], state['request_sha256'])
            page = request['pages'][0]
            self.assertEqual(page['source_name'], '中文页.png')
            self.assertEqual(archive.read(page['file']), (self.images/'中文页.png').read_bytes())
            self.assertEqual(page['options'], {'mask_dilate_size': 7})

    def test_cbz_nested_name(self):
        source = self.root/'comic.cbz'
        with zipfile.ZipFile(source, 'w') as archive:
            archive.writestr('漫画/页.png', png())
            archive.writestr('readme.txt', 'ignored')
        state = client.prepare(source, self.root/'cbz-job')
        self.assertEqual(state['pages'], 1)
        request = client.load_json(self.root/'cbz-job/request.json')
        self.assertEqual(request['pages'][0]['source_name'], '漫画/页.png')

    def test_unsafe_archive_rejected(self):
        source = self.root/'bad.cbz'
        with zipfile.ZipFile(source, 'w') as archive:
            archive.writestr('../evil.png', png())
        with self.assertRaises(ValueError):
            client.prepare(source, self.root/'bad-job')

    def test_unknown_page_override_rejected(self):
        with self.assertRaises(ValueError):
            self.prepare({'pages': {'missing.png': {}}})

    def test_malformed_page_options(self):
        self.prepare()
        request = client.load_json(self.folder/'request.json')
        request['pages'][0]['options'] = []
        with self.assertRaises(ValueError):
            worker.validate_request(request)

    def test_manifest_checksum_rejected(self):
        self.prepare()
        with zipfile.ZipFile(self.folder/'task.bin') as archive, self.assertRaises(ValueError):
            worker.read_task(archive, expected_request_sha='0'*64)

    def test_duplicate_archive_member_rejected(self):
        data = io.BytesIO()
        with zipfile.ZipFile(data, 'w') as archive:
            archive.writestr('request.json', '{}')
            with self.assertWarns(UserWarning):
                archive.writestr('request.json', '{}')
        with zipfile.ZipFile(data) as archive, self.assertRaises(ValueError):
            worker.read_task(archive)

    def test_empty_input_rejected(self):
        empty = self.root/'empty'
        empty.mkdir()
        with self.assertRaises(ValueError):
            client.prepare(empty, self.root/'empty-job')

    def test_invalid_image_rejected(self):
        with self.assertRaises(OSError):
            worker.decode_image(b'not an image')

    def test_completed_output_verified_and_gray_mask_preserved(self):
        result = self.run_job()
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(client.verify_output(self.root/'outputs', self.state),
                         json.loads(worker.json_bytes(result)))
        with zipfile.ZipFile(self.root/'outputs/results.zip') as archive:
            with worker.Image.open(io.BytesIO(archive.read('p000001/precise-mask.png'))) as image:
                self.assertEqual(image.getextrema(), (17, 17))

    def test_page_error_is_explicit(self):
        engine = Mock()
        engine.process.side_effect = RuntimeError('out of memory')
        result = self.run_job(engine)
        self.assertEqual(result['status'], 'failed')
        self.assertIn('out of memory', result['pages'][0]['error'])
        client.verify_output(self.root/'outputs', self.state)

    def test_partial_task_retains_successful_pages(self):
        (self.images/'second.png').write_bytes(png('black'))
        engine = Mock()
        engine.process.side_effect = [FakeEngine().process(worker.Image.new('RGB',(12,18))),
                                      RuntimeError('one page failed')]
        result = self.run_job(engine)
        self.assertEqual((result['status'], result['succeeded'], result['failed']), ('partial', 1, 1))
        client.verify_output(self.root/'outputs', self.state)

    def test_output_budget_does_not_emit_partial_page_files(self):
        result = self.run_job(output_limit=1)
        self.assertEqual(result['status'], 'failed')
        with zipfile.ZipFile(self.root/'outputs/results.zip') as archive:
            self.assertEqual(set(archive.namelist()), {'job-result.json', 'p000001/result.json'})

    def test_wrong_job_and_runtime_rejected(self):
        self.run_job()
        for key in ('job_id', 'runtime_sha256'):
            state = {**self.state, key: 'wrong'}
            with self.subTest(key=key), self.assertRaises(ValueError):
                client.verify_output(self.root/'outputs', state)

    def test_download_tampering_rejected(self):
        self.run_job()
        with (self.root/'outputs/results.zip').open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaises(ValueError):
            client.verify_output(self.root/'outputs', self.state)

    def test_notebook_contains_only_config_and_worker(self):
        self.prepare()
        script = client.make_script({'runtime_sha256': 'a'*64}, self.state)
        self.assertIn(self.state['job_id'], script)
        self.assertNotIn('KGAT_', script)
        self.assertNotIn('KAGGLE_API_TOKEN', script)
        compile(script, '<job>', 'exec')

    def test_cleanup_dry_run_and_scope(self):
        api = Mock()
        api.config_values = {'username': 'owner'}
        state = dict(job_id='lama-test-12345678', notebook='owner/lama-test-12345678',
            input_dataset='owner/lama-test-12345678-input', runtime_dataset='owner/runtime')
        client.cleanup(api, state)
        api.kernels_delete.assert_not_called()
        api.dataset_delete.assert_not_called()
        state['input_dataset'] = 'owner/runtime'
        with self.assertRaises(ValueError):
            client.cleanup(api, state, yes=True)

    def test_cleanup_running_task_rejected(self):
        api = Mock()
        api.config_values = {'username': 'owner'}
        api.kernels_status.return_value.status = 'running'
        state = dict(job_id='lama-test-12345678', notebook='owner/lama-test-12345678',
            input_dataset='owner/lama-test-12345678-input', runtime_dataset='owner/runtime')
        with self.assertRaises(ValueError):
            client.cleanup(api, state, yes=True)
        api.kernels_delete.assert_not_called()

    def test_sdk_enum_status(self):
        class Status(enum.Enum):
            COMPLETE = 2
        api = Mock()
        api.kernels_status.return_value.status = Status.COMPLETE
        api.kernels_status.return_value.failure_message = ''
        self.assertEqual(client.kernel_status(api, {'notebook':'owner/job'}), ('complete', ''))

    def test_fetch_uses_explicit_paging_and_utf8_logs(self):
        self.run_job()
        api = Mock()
        api.kernels_output.return_value = ([], '')
        api.kernels_logs.return_value = '中文 ✅'
        def download(ref, **kwargs):
            import shutil
            for name in ('job-result.json', 'results.zip'):
                shutil.copyfile(self.root/'outputs'/name, Path(kwargs['path'])/name)
            return [], ''
        api.kernels_output.side_effect = download
        self.state['notebook'] = 'owner/job'
        client.fetch(api, self.state, self.root/'download')
        self.assertEqual(api.kernels_output.call_args.kwargs['page_token'], '')
        self.assertEqual((self.root/'download/notebook.log').read_text(encoding='utf-8'), '中文 ✅')


if __name__ == '__main__':
    unittest.main()
