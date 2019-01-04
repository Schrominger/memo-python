# -*- coding: utf-8 -*-
import os
import sys
import paramiko


class mySSH(object):

	def __init__(self, ip, usr, passwd, **kwargs):
		self.ip = ip
		self.usr = usr
		self.passwd = passwd
		self.port = kwargs.get('port',22)
		self.ssh = None

	def ssh_connect(self, *args, **kwargs):
	    self.ssh = paramiko.SSHClient()
	    self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
	    # ssh connnection
	    self.ssh.connect(self.ip, username=self.usr, password=self.passwd, port=self.port)


	def send_file(self, localFile, remotePath, *args, **kwargs):
	    # trasnport one file to remote
		remoteFile = os.path.join(remotePath, localFile.split('/')[-1])

		if self.ssh is None:
			# sftp
			self.ssh_connect()
			sftp = self.ssh.open_sftp()
		else:
			sftp = self.ssh.open_sftp()

		print('Sending file to remote...')
		sftp.put(localFile, remoteFile)
		print('Complete send!')
		sftp.close()


	def send_folder_files(self, localPath, remotePath, *args, **kwargs):
		"""
		trasnport all files in given folder to remote folder
		"""
		if self.ssh is None:
			# sftp
			self.ssh_connect()
			sftp = self.ssh.open_sftp()
		else:
			sftp = self.ssh.open_sftp()

		for root, dirs, files in os.walk(localPath):
			for fname in files:
				full_fname = os.path.join(root, fname)
				print (u'Put file %s tansporting...' % fname)
				sftp.put(full_fname, os.path.join(remotePath, fname))

		sftp.close()

	def get_file(self, remotefile, localPath):
		if self.ssh is None:
			# sftp
			self.ssh_connect()
			sftp = self.ssh.open_sftp()
		else:
			sftp = self.ssh.open_sftp()

		localfile = os.path.join(localPath, remotefile.split('/')[-1])
		print('Get file %s from remote...' %remotefile.split('/')[-1])
		sftp.get(remotefile, localfile)
		sftp.close()

	def get_all_files(self, localPath, remotePath, *args, **kwargs):
		"""get all files in remote folder"""
		if not os.path.exists(localPath):
			os.makedirs(localPath)

		if self.ssh is None:
			self.ssh_connect()
			sftp = self.ssh.open_sftp()
		else:
			sftp = self.ssh.open_sftp()

		remote_files = sftp.listdir(remotePath)
		print(remote_files)
		try:
			for file in remote_files:
				local_file = os.path.join(localPath,file)
				remote_file = os.path.join(remotePath,file)
				print('Get file %s from remote...' %file)
				sftp.get(remote_file, local_file)
		except IOError:   
			return ("remote_path or local_path not exist")

		sftp.close()

	def ssh_remote_exec(self, cmd, *args,**kwargs):

		if self.ssh is None:
			self.ssh_connect()
		print('execute on remote: %s' %cmd)
		stdin,stdout,stderr = self.ssh.exec_command(cmd)

		for line in stdout:
			# Process each line in the remote output
			print (line)
		return stdin,stdout,stderr

	def close(self,):
		if self.ssh is not None:
			self.ssh.close()
		print('close ssh connect.')

