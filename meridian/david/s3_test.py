import os
from pathlib import Path
from absl.testing import absltest

from meridian.david import s3

mock = absltest.mock


class PushAndPurgeTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.tmp = self.create_tempdir()
    self.file_name = 'file.txt'
    with open(os.path.join(self.tmp.full_path, self.file_name), 'w') as f:
      f.write('data')

  def test_uploads_and_removes_dir(self):
    client = mock.Mock()
    with (
        mock.patch.object(s3.boto3, 'client', return_value=client) as m_client,
        mock.patch.object(s3.shutil, 'rmtree') as m_rmtree,
        mock.patch('builtins.print') as m_print,
    ):
      s3.push_and_purge(
          self.tmp.full_path,
          self.file_name,
          bucket='bucket',
          team_prefix='team/',
          user_space='user',
          file_structure='fs',
      )

    m_client.assert_called_once_with('s3')
    expected_local = Path(self.tmp.full_path) / self.file_name
    expected_key = 'team/user/fs/' + self.file_name
    client.upload_file.assert_called_once_with(
        str(expected_local), 'bucket', expected_key)
    m_rmtree.assert_called_once_with(Path(self.tmp.full_path), ignore_errors=True)
    m_print.assert_any_call(
        f'Uploaded {expected_local} to s3://bucket/{expected_key}')
    m_print.assert_any_call(
        f'Removed temporary directory {self.tmp.full_path}')

  def test_upload_error_still_removes_dir(self):
    err = RuntimeError('boom')
    client = mock.Mock()
    client.upload_file.side_effect = err
    with (
        mock.patch.object(s3.boto3, 'client', return_value=client),
        mock.patch.object(s3.shutil, 'rmtree') as m_rmtree,
        mock.patch.object(s3.traceback, 'print_exc') as m_exc,
    ):
      with self.assertRaises(RuntimeError):
        s3.push_and_purge(
            self.tmp.full_path,
            self.file_name,
            bucket='b',
            team_prefix='t',
            user_space='u',
            file_structure='f',
        )
    m_rmtree.assert_called_once_with(Path(self.tmp.full_path), ignore_errors=True)
    m_exc.assert_called_once()


class PullLoadPurgeTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.tmp = self.create_tempdir()

  def test_downloads_and_cleans_up(self):
    client = mock.Mock()
    loader = mock.Mock(return_value='obj')
    with (
        mock.patch.object(s3.boto3, 'client', return_value=client) as m_client,
        mock.patch('pathlib.Path.unlink') as m_unlink,
        mock.patch('pathlib.Path.rmdir') as m_rmdir,
        mock.patch.object(s3.shutil, 'rmtree') as m_rmtree,
        mock.patch('builtins.print') as m_print,
    ):
      result = s3.pull_load_purge(
          's3://b/prefix/file.txt',
          loader,
          tmp_dir=self.tmp.full_path,
      )

    m_client.assert_called_once_with('s3')
    expected_local = Path(self.tmp.full_path) / 'file.txt'
    client.download_file.assert_called_once_with(
        'b',
        'prefix/file.txt',
        str(expected_local),
    )
    loader.assert_called_once_with(expected_local)
    m_unlink.assert_called_once_with(missing_ok=True)
    m_rmdir.assert_called_once_with()
    m_rmtree.assert_not_called()
    m_print.assert_any_call(
        f'Downloaded s3://b/prefix/file.txt to {expected_local}')
    m_print.assert_any_call('Object loaded.')
    m_print.assert_any_call(
        f'Removed temporary directory {self.tmp.full_path}')
    self.assertEqual(result, 'obj')

  def test_download_error_still_cleans_up(self):
    err = RuntimeError('boom')
    client = mock.Mock()
    client.download_file.side_effect = err
    loader = mock.Mock()
    with (
        mock.patch.object(s3.boto3, 'client', return_value=client),
        mock.patch('pathlib.Path.unlink') as m_unlink,
        mock.patch('pathlib.Path.rmdir', side_effect=OSError) as m_rmdir,
        mock.patch.object(s3.shutil, 'rmtree') as m_rmtree,
        mock.patch.object(s3.traceback, 'print_exc') as m_exc,
    ):
      with self.assertRaises(RuntimeError):
        s3.pull_load_purge(
            's3://b/prefix/file.txt',
            loader,
            tmp_dir=self.tmp.full_path,
        )

    m_unlink.assert_called_once_with(missing_ok=True)
    m_rmdir.assert_called_once_with()
    m_rmtree.assert_called_once_with(Path(self.tmp.full_path), ignore_errors=True)
    m_exc.assert_called_once()


class GeneratePresignedUrlTest(absltest.TestCase):

  def test_uses_default_client(self):
    client = mock.Mock()
    client.generate_presigned_url.return_value = 'url'
    with mock.patch.object(s3.boto3, 'client', return_value=client) as m_client:
      url = s3.generate_presigned_url('b', 'k', expires_in=10)
    m_client.assert_called_once_with('s3')
    client.generate_presigned_url.assert_called_once_with(
        ClientMethod='get_object',
        Params={'Bucket': 'b', 'Key': 'k'},
        ExpiresIn=10,
    )
    self.assertEqual(url, 'url')

  def test_uses_provided_client(self):
    client = mock.Mock()
    client.generate_presigned_url.return_value = 'u'
    url = s3.generate_presigned_url('b', 'k', expires_in=5, s3_client=client)
    client.generate_presigned_url.assert_called_once_with(
        ClientMethod='get_object',
        Params={'Bucket': 'b', 'Key': 'k'},
        ExpiresIn=5,
    )
    self.assertEqual(url, 'u')


class DisplayHtmlLinkTest(absltest.TestCase):

  def test_returns_html(self):
    with mock.patch.object(s3, 'generate_presigned_url', return_value='http://l') as m_gen, \
         mock.patch('IPython.display.HTML') as MockHTML:
      result = s3.display_html_link('b', 'k', expires_in=3)
    m_gen.assert_called_once_with(bucket='b', key='k', expires_in=3)
    MockHTML.assert_called_once_with(
        '<a href="http://l" target="_blank">Open report in new tab</a>')
    self.assertIs(result, MockHTML.return_value)


class AllExportedTest(absltest.TestCase):

  def test_expected_exports(self):
    self.assertCountEqual(
        s3.__all__,
        [
            'push_and_purge',
            'pull_load_purge',
            'generate_presigned_url',
            'display_html_link',
        ],
    )


if __name__ == '__main__':
  absltest.main()
