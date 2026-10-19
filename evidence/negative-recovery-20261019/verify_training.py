"""Read checkpoint metadata/scalars and training logs only; never load model weights."""
from pathlib import Path
import hashlib,json,re
import numpy as np
import tensorstore as ts
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/home/robot/.pi05-handson')
OUT=ROOT/'evidence/negative-recovery-20261019'
LOG=ROOT/'logs/ui_training_1792390504392667072.log'
LAUNCH=ROOT/'evidence/negative_quadrants_20261019-launch.json'
CHECKPOINT=ROOT/'checkpoints/pi05_handson_lora/negative_quadrants_20261019/1999'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def scalar(path,name,zarr3):
 return int(ts.open({'driver':'zarr3' if zarr3 else 'zarr','kvstore':{'driver':'ocdbt','base':{'driver':'file','path':str(path)},'path':name},'fill_missing_data_reads':False},open=True).result().read().result())
text=LOG.read_text();launch=json.loads(LAUNCH.read_text())
patterns=[r'Finished asynchronous save[^\n]+/1999\b',r'\[step=1999\]\[wait_until_finished\] Done waiting for Save Finalize']
completion=[next((x for x in text.splitlines() if re.search(pattern,x)),None) for pattern in patterns]
assert all(completion),'Final asynchronous save has not completed'
assert CHECKPOINT.is_dir() and not list(CHECKPOINT.parent.glob('1999.*tmp*'))
raw=re.findall(r'Step (\d+): grad_norm=([^,\n]+), loss=([^,\n]+), param_norm=([^\s\n]+)',text)
steps=np.array([int(r[0]) for r in raw]);values=np.array([[float(x) for x in r[1:]] for r in raw])
assert steps.tolist()==list(range(0,2000,10));assert np.isfinite(values).all()
assert not re.search(r'Traceback \(most recent call last\)',text)
assert launch['new_updates']==2000 and launch['batch_size']==4 and launch['seed']==42
assert launch['discrete_state_input'] is False and launch['action_horizon']==10 and launch['dry_run'] is False
assert launch['learning_rate']['peak']==launch['learning_rate']['final']==1e-5
manifest=Path(launch['dataset_manifest']);assert sha(manifest)==launch['dataset_manifest_sha256']
parent=Path(launch['parent_checkpoint']);assert parent.name=='1999' and parent.parent.name=='retrain_teacher_1x_20261019'
assert sha(parent/'params/manifest.ocdbt')==launch['parent_params_manifest_sha256']
assert f'Restoring checkpoint from {parent}/params.' in text
for component in ['params','train_state','assets']:assert (CHECKPOINT/component).is_dir()
for component in ['params','train_state']:
 for filename in ['manifest.ocdbt','_METADATA']:assert (CHECKPOINT/component/filename).stat().st_size>0
 assert any(p.is_file() and p.stat().st_size for p in (CHECKPOINT/component/'d').rglob('*'))
state_metadata=json.loads((CHECKPOINT/'train_state/_METADATA').read_text())
step=scalar(CHECKPOINT/'train_state','step',state_metadata['use_zarr3']);assert step==2000
moments={'mu':[],'nu':[]};counts={}
for entry in state_metadata['tree_metadata'].values():
 keys=[x['key'] for x in entry['key_metadata']]
 for moment in moments:
  if moment in keys and entry['value_metadata']['value_type']=='jax.Array':moments[moment].append(keys[keys.index(moment)+1:])
 if 'count' in keys:
  name='.'.join(keys);counts[name]=scalar(CHECKPOINT/'train_state',name,state_metadata['use_zarr3'])
assert len(moments['mu'])==len(moments['nu'])==20
assert moments['mu']==moments['nu'] and all(any('lora' in key for key in path) for path in moments['mu'])
assert counts and set(counts.values())=={2000}
assert sha(ROOT/'scripts/course_config.py')==launch['source_sha256']['course_config.py']
norm=CHECKPOINT/'assets/course/libero_black_bowl/norm_stats.json';assert sha(norm)==launch['normalization']['sha256'];assert sha(norm)==sha(parent/'assets/course/libero_black_bowl/norm_stats.json')==sha(ROOT/launch['normalization']['path'])
loss=values[:,1];fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
ax.plot(steps,loss,color='#789ac7',linewidth=1,alpha=.8,label='Logged training loss')
ax.plot(steps[9:],np.convolve(loss,np.ones(10)/10,mode='valid'),color='#154b83',linewidth=2,label='Trailing mean of 10 logged values')
ax.set(xlabel='Training update index',ylabel='Training loss',title='pi0.5 LoRA: negative-position recovery training')
ax.grid(alpha=.2);ax.legend(frameon=False)
fig.supxlabel('Training objective only; this is not a task success rate. Logs summarize windows of up to 10 updates.',fontsize=9)
fig.savefig(OUT/'training-loss.png',dpi=200);plt.close(fig)
metadata_files=['_CHECKPOINT_METADATA','params/manifest.ocdbt','params/_METADATA','train_state/manifest.ocdbt','train_state/_METADATA','assets/course/libero_black_bowl/norm_stats.json']
report={'status':'PASS','experiment':'negative_quadrants_20261019','log':str(LOG.relative_to(ROOT)),'log_sha256':sha(LOG),'launch_manifest':str(LAUNCH.relative_to(ROOT)),'launch_manifest_sha256':sha(LAUNCH),'configuration':{k:launch[k] for k in ['new_updates','batch_size','seed','action_horizon','discrete_state_input','learning_rate','optimizer','lineage','parent_checkpoint','parent_params_manifest_sha256','dataset_manifest','dataset_manifest_sha256','repo_id','source_commit','openpi_commit','base_holdout']},'sampling':launch['recovery_sampling'],'logged_statistics':{'first_index':int(steps[0]),'last_index':int(steps[-1]),'count':len(steps),'expected_stride':10,'all_expected_indices_present_once':True,'all_loss_grad_norm_param_norm_finite':True,'first_loss':float(loss[0]),'last_logged_loss':float(loss[-1]),'min_loss':float(loss.min()),'max_loss':float(loss.max()),'first20_logged_mean_loss':float(loss[:20].mean()),'last20_logged_mean_loss':float(loss[-20:].mean()),'meaning':'Step0 summarizes its first update; subsequent logged values average the preceding10 training updates. Last nine updates are not separately printed.'},'checkpoint':{'directory':str(CHECKPOINT.relative_to(ROOT)),'stored_train_state_step':step,'optimizer_schedule_count_scalars':counts,'metadata_and_norm_sha256':{name:sha(CHECKPOINT/name) for name in metadata_files},'components_present':['params','train_state','assets'],'asynchronous_save_completion_log_lines':completion,'norm_matches_parent_and_course':True},'lora_only_evidence':{'optimizer_mu_tensor_count':20,'optimizer_nu_tensor_count':20,'all_optimizer_moment_parameter_paths_contain_lora':True,'trainable_tensor_paths':moments['mu'],'course_config_sha256_matches_launch':True,'filter_source':'scripts/course_config.py freezes nnx.Param except PathRegex(.*lora.*); trainer differentiates/updates config.trainable_filter only.'},'loss_plot':{'path':'evidence/negative-recovery-20261019/training-loss.png','sha256':sha(OUT/'training-loss.png'),'method':'Matplotlib; logged means and trailing10logged-value mean, no invented validation curve.'},'limitations':['Training loss is not rollout success, release quality or recovery/generalization ability; policy evaluation is handled separately.','No model parameters or optimizer moment arrays were loaded. Only scalar counters and serialized optimizer path metadata were read.','Frozen weights were not independently compared byte-for-byte with the parent; LoRA-only evidence is the pinned freeze filter and saved optimizer tensor inventory.','No subprocess exit code is claimed by this read-only audit; final async-save completion and saved2000update counters establish checkpoint completion.']}
(OUT/'training-verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':'PASS','stored_step':step,'optimizer_counts':counts,'logged_count':len(steps),'last_loss':float(loss[-1]),'first20mean':float(loss[:20].mean()),'last20mean':float(loss[-20:].mean())}))
