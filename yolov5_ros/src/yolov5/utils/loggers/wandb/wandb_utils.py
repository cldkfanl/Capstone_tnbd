"""Utilities and tools for tracking runs with Weights & Biases."""

import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict

import yaml
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[3]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from utils.dataloaders import LoadImagesAndLabels, img2label_paths
from utils.general import LOGGER, check_dataset, check_file

try:
    import wandb

    assert hasattr(wandb, '__version__')  # verify package import not local dir
except (ImportError, AssertionError):
    wandb = None

RANK = int(os.getenv('RANK', -1))
WANDB_ARTIFACT_PREFIX = 'wandb-artifact://'


def remove_prefix(from_string, prefix=WANDB_ARTIFACT_PREFIX):
    return from_string[len(prefix):]


def check_wandb_config_file(data_config_file):
    wandb_config = '_wandb.'.join(data_config_file.rsplit('.', 1))  # updated data.yaml path
    if Path(wandb_config).is_file():
        return wandb_config
    return data_config_file


def check_wandb_dataset(data_file):
    is_trainset_wandb_artifact = False
    is_valset_wandb_artifact = False
    if isinstance(data_file, dict):
        # In that case another dataset manager has already processed it and we don't have to
        return data_file
    if check_file(data_file) and data_file.endswith('.yaml'):
        with open(data_file, errors='ignore') as f:
            data_dict = yaml.safe_load(f)
        is_trainset_wandb_artifact = isinstance(data_dict['train'],
                                                str) and data_dict['train'].startswith(WANDB_ARTIFACT_PREFIX)
        is_valset_wandb_artifact = isinstance(data_dict['val'],
                                              str) and data_dict['val'].startswith(WANDB_ARTIFACT_PREFIX)
    if is_trainset_wandb_artifact or is_valset_wandb_artifact:
        return data_dict
    else:
        return check_dataset(data_file)


def get_run_info(run_path):
    run_path = Path(remove_prefix(run_path, WANDB_ARTIFACT_PREFIX))
    run_id = run_path.stem
    project = run_path.parent.stem
    entity = run_path.parent.parent.stem
    model_artifact_name = 'run_' + run_id + '_model'
    return entity, project, run_id, model_artifact_name


def check_wandb_resume(opt):
    process_wandb_config_ddp_mode(opt) if RANK not in [-1, 0] else None
    if isinstance(opt.resume, str):
        if opt.resume.startswith(WANDB_ARTIFACT_PREFIX):
            if RANK not in [-1, 0]:  # For resuming DDP runs
                entity, project, run_id, model_artifact_name = get_run_info(opt.resume)
                api = wandb.Api()
                artifact = api.artifact(entity + '/' + project + '/' + model_artifact_name + ':latest')
                modeldir = artifact.download()
                opt.weights = str(Path(modeldir) / "last.pt")
            return True
    return None


def process_wandb_config_ddp_mode(opt):
    with open(check_file(opt.data), errors='ignore') as f:
        data_dict = yaml.safe_load(f)  # data dict
    train_dir, val_dir = None, None
    if isinstance(data_dict['train'], str) and data_dict['train'].startswith(WANDB_ARTIFACT_PREFIX):
        api = wandb.Api()
        train_artifact = api.artifact(remove_prefix(data_dict['train']) + ':' + opt.artifact_alias)
        train_dir = train_artifact.download()
        train_path = Path(train_dir) / 'data/images/'
        data_dict['train'] = str(train_path)

    if isinstance(data_dict['val'], str) and data_dict['val'].startswith(WANDB_ARTIFACT_PREFIX):
        api = wandb.Api()
        val_artifact = api.artifact(remove_prefix(data_dict['val']) + ':' + opt.artifact_alias)
        val_dir = val_artifact.download()
        val_path = Path(val_dir) / 'data/images/'
        data_dict['val'] = str(val_path)
    if train_dir or val_dir:
        ddp_data_path = str(Path(val_dir) / 'wandb_local_data.yaml')
        with open(ddp_data_path, 'w') as f:
            yaml.safe_dump(data_dict, f)
        opt.data = ddp_data_path


class WandbLogger():
    """Log training runs, datasets, models, and predictions to Weights & Biases.

    This logger sends information to W&B at wandb.ai. By default, this information
    includes hyperparameters, system configuration and metrics, model metrics,
    and basic data metrics and analyses.

    By providing additional command line arguments to train.py, datasets,
    models and predictions can also be logged.

    For more on how this logger is used, see the Weights & Biases documentation:
    https://docs.wandb.com/guides/integrations/yolov5
    """

    def __init__(self, opt, run_id=None, job_type='Training'):
        """
        - Initialize WandbLogger instance
        - Upload dataset if opt.upload_dataset is True
        - Setup training processes if job_type is 'Training'

        arguments:
        opt (namespace) -- Commandline arguments for this run
        run_id (str) -- Run ID of W&B run to be resumed
        job_type (str) -- To set the job_type for this run

       """
        # Pre-training routine --
        self.job_type = job_type
        self.wandb, self.wandb_run = wandb, None if not wandb else wandb.run
        self.val_artifact, self.train_artifact = None, None
        self.train_artifact_path, self.val_artifact_path = None, None
        self.result_artifact = None
        self.val_table, self.result_table = None, None
        self.bbox_media_panel_images = []
        self.val_table_path_map = None
        self.max_imgs_to_log = 16
        self.wandb_artifact_data_dict = None
        self.data_dict = None
        # It's more elegant to stick to 1 wandb.init call,
        #  but useful config data is overwritten in the WandbLogger's wandb.init call
        if isinstance(opt.resume, str):  # checks resume from artifact
            if opt.resume.startswith(WANDB_ARTIFACT_PREFIX):
                entity, project, run_id, model_artifact_name = get_run_info(opt.resume)
                model_artifact_name = WANDB_ARTIFACT_PREFIX + model_artifact_name
                assert wandb, 'install wandb to resume wandb runs'
                # Resume wandb-artifact:// runs here| workaround for not overwriting wandb.config
                self.wandb_run = wandb.init(id=run_id,
                                            project=project,
                                            entity=entity,
                                            resume='allow',
                                            allow_val_change=True)
                opt.resume = model_artifact_name
        elif self.wandb:
            self.wandb_run = wandb.init(config=opt,
                                        resume="allow",
                                        project='YOLOv5' if opt.project == 'runs/train' else Path(opt.project).stem,
                                        entity=opt.entity,
                                        name=opt.name if opt.name != 'exp' else None,
                                        job_type=job_type,
                                        id=run_id,
                                        allow_val_change=True) if not wandb.run else wandb.run
        if self.wandb_run:
            if self.job_type == 'Training':
                if opt.upload_dataset:
                    if not opt.resume:
                        self.wandb_artifact_data_dict = self.check_and_upload_dataset(opt)

                if isinstance(opt.data, dict):
                    # This means another dataset manager has already processed the dataset info (e.g. ClearML)
                    # and they will have stored the already processed dict in opt.data
                    self.data_dict = opt.data
                elif opt.resume:
                    # resume from artifact
                    if isinstance(opt.resume, str) and opt.resume.startswith(WANDB_ARTIFACT_PREFIX):
                        self.data_dict = dict(self.wandb_run.config.data_dict)
                    else:  # local resume
                        self.data_dict = check_wandb_dataset(opt.data)
                else:
                    self.data_dict = check_wandb_dataset(opt.data)
                    self.wandb_artifact_data_dict = self.wandb_artifact_data_dict or self.data_dict

                    # write data_dict to config. useful for resuming from artifacts. Do this only when not resuming.
                    self.wandb_run.config.update({'data_dict': self.wandb_artifact_data_dict}, allow_val_change=True)
                self.setup_training(opt)

            if self.job_type == 'Dataset Creation':
                self.wandb_run.config.update({"upload_dataset": True})
                self.data_dict = self.check_and_upload_dataset(opt)

    def check_and_upload_dataset(self, opt):
        """
        Check if the dataset format is compatible and upload it as W&B artifact

        arguments:
        opt (namespace)-- Commandline arguments for current run

        returns:
        Updated dataset info dictionary where local dataset paths are replaced by WAND_ARFACT_PREFIX links.
        """
        assert wandb, 'Install wandb to upload dataset'
        config_path = self.log_dataset_artifact(opt.data, opt.single_cls,
                                                'YOLOv5' if opt.project == 'runs/train' else Path(opt.project).stem)
        with open(config_path, errors='ignore') as f:
            wandb_data_dict = yaml.safe_load(f)
        return wandb_data_dict

    def setup_training(self, opt):
        """
        Setup the necessary processes for training YOLO models:
          - Attempt to download model checkpoint and dataset artifacts if opt.resume stats with WANDB_ARTIFACT_PREFIX
          - Update data_dict, to contain info of previous run if resumed and the paths of dataset artifact if downloaded
          - Setup log_dict, initialize bbox_interval

        arguments:
        opt (namespace) -- commandline arguments for this run

        """
        self.log_dict, self.current_epoch = {}, 0
        self.bbox_interval = opt.bbox_interval
        if isinstance(opt.resume, str):
            modeldir, _ = self.download_model_artifact(opt)
            if modeldir:
                self.weights = Path(modeldir) / "last.pt"
                config = self.wandb_run.config
                opt.weights, opt.save_period, opt.batch_size, opt.bbox_interval, opt.epochs, opt.hyp, opt.imgsz = str(
                    self.weights), config.save_period, config.batch_size, config.bbox_interval, config.epochs,\
                    config.hyp, config.imgsz
        data_dict = self.data_dict
        if self.val_artifact is None:  # If --upload_dataset is set, use the existing artifact, don't download
            self.train_artifact_path, self.train_artifact = self.download_dataset_artifact(
                data_dict.get('train'), opt.artifact_alias)
            self.val_artifact_path, self.val_artifact = self.download_dataset_artifact(
                data_dict.get('val'), opt.artifact_alias)

        if self.train_artifact_path is not None:
            train_path = Path(self.train_artifact_path) / 'data/images/'
            data_dict['train'] = str(train_path)
        if self.val_artifact_path is not None:
            val_path = Path(self.val_artifact_path) / 'data/images/'
            data_dict['val'] = str(val_path)

        if self.val_artifact is not None:
            self.result_artifact = wandb.Artifact("run_" + wandb.run.id + "_progress", "evaluation")
            columns = ["epoch", "id", "ground truth", "prediction"]
            columns.extend(self.data_dict['names'])
            self.result_table = wandb.Table(columns)
            self.val_table = self.val_artifact.get("val")
            if self.val_table_path_map is None:
                self.map_val_table_path()
        if opt.bbox_interval == -1:
            self.bbox_interval = opt.bbox_interval = (opt.epochs // 10) if opt.epochs > 10 else 1
            if opt.evolve or opt.noplots:
                self.bbox_interval = opt.bbox_interval = opt.epochs + 1  # disable bbox_interval
        train_from_artifact = self.train_artifact_path is not None and self.val_artifact_path is not None
        # Update the the data_dict to point to local artifacts dir
        if train_from_artifact:
            self.data_dict = data_dict

    def download_dataset_artifact(self, path, alias):
        """
        download the model checkpoint artifact if the path starts with WANDB_ARTIFACT_PREFIX

        arguments:
        path -- path of the dataset to be used for training
        alias (str)-- alias of the artifact to be download/used for training

        returns:
        (str, wandb.Artifact) -- path of the downladed dataset and it's corresponding artifact object if dataset
        is found otherwise returns (None, None)
        """
        if isinstance(path, str) and path.startswith(WANDB_ARTIFACT_PREFIX):
            artifact_path = Path(remove_prefix(path, WANDB_ARTIFACT_PREFIX) + ":" + alias)
            dataset_artifact = wandb.use_artifact(artifact_path.as_posix().replace("\\", "/"))
            assert dataset_artifact is not None, "'Error: W&B dataset artifact doesn\'t exist'"
            datadir = dataset_artifact.download()
            return datadir, dataset_artifact
        return None, None

    def download_model_artifact(self, opt):
        """
        download the model checkpoint artifact if the resume path starts with WANDB_ARTIFACT_PREFIX

        arguments:
        opt (namespace) -- Commandline arguments for this run
        """
        if opt.resume.startswith(WANDB_ARTIFACT_PREFIX):
            model_artifact = wandb.use_artifact(remove_prefix(opt.resume, WANDB_ARTIFACT_PREFIX) + ":latest")
            assert model_artifact is not None, 'Error: W&B model artifact doesn\'t exist'
            modeldir = model_artifact.download()
            # epochs_trained = model_artifact.metadata.get('epochs_trained')
            total_epochs = model_artifact.metadata.get('total_epochs')
            is_finished = total_epochs is None
            assert not is_finished, 'training is finished, can only resume incomplete runs.'
            return modeldir, model_artifact
        return None, None

    def log_model(self, path, opt, epoch, fitness_score, best_model=False):
        """
        Log the model checkpoint as W&B artifact

        arguments:
        path (Path)   -- Path of directory containing the checkpoints
        opt (namespace) -- Command line arguments for this run
        epoch (int)  -- Current epoch number
        fitness_score (float) -- fitness score for current epoch
        best_model (boolean) -- Boolean representing if the current checkpoint is the best yet.
        """
        model_artifact = wandb.Artifact('run_' + wandb.run.id + '_model',
                                        type='model',
                                        metadata={
                                            'original_url': str(path),
                                            'epochs_trained': epoch + 1,
                                            'save period': opt.save_period,
                                            'project': opt.project,
                                            'total_epochs': opt.epochs,
                                            'fitness_score': fitness_score})
        model_artifact.add_file(str(path / 'last.pt'), name='last.pt')
        wandb.log_artifact(model_artifact,
                           aliases=['latest', 'last', 'epoch ' + str(self.current_epoch), 'best' if best_model else ''])
        LOGGER.info(f"Saving model artifact on epoch {epoch + 1}")

    def log_dataset_artifact(self, data_file, single_cls, project, overwrite_config=False):
        """
        Log the dataset as W&B artifact and return the new data file with W&B links

        arguments:
        data_file (str) -- the .yaml file with information about the dataset like - path, classes etc.
        single_class (boolean)  -- train multi-class data as single-class
        project (str) -- project name. Used to construct the artifact path
        overwrite_config (boolean) -- overwrites the data.yaml file if set to true otherwise creates a new
        file with _wandb postfix. Eg -> data_wandb.yaml

        returns:
        the new .yaml file with artifact links. it can be used to start training directly from artifacts
        """
        upload_dataset = self.wandb_run.config.upload_dataset
        log_val_only = isinstance(upload_dataset, str) and upload_dataset == 'val'
        self.data_dict = check_dataset(data_file)  # parse and check
        data = dict(self.data_dict)
        nc, names = (1, ['item']) if single_cls else (int(data['nc']), data['names'])
        names = {k: v for k, v in enumerate(names)}  # to index dictionary

        # log train set
        if not log_val_only:
            self.train_artifact = self.create_dataset_table(LoadImagesAndLabels(data['train'], rect=True, batch_size=1),
                                                            names,
                                                            name='train') if data.get('train') else None
            if data.get('train'):
                data['train'] = WANDB_ARTIFACT_PREFIX + str(Path(project) / 'train')

        self.val_artifact = self.create_dataset_table(
            LoadImagesAndLabels(data['val'], rect=True, batch_size=1), names, name='val') if data.get('val') else None
        if data.get('val'):
            data['val'] = WANDB_ARTIFACT_PREFIX + str(Path(project) / 'val')

        path = Path(data_file)
        # create a _wandb.yaml file with artifacts links if both train and test set are logged
        if not log_val_only:
            path = (path.stem if overwrite_config else path.stem + '_wandb') + '.yaml'  # updated data.yaml path
            path = ROOT / 'data' / path
            data.pop('download', None)
            data.pop('path', None)
            with open(path, 'w') as f:
                yaml.safe_dump(data, f)
                LOGGER.info(f"Created dataset config file {path}")

        if self.job_type == 'Training':  # builds correct artifact pipeline graph
            if not log_val_only:
                self.wandb_run.log_artifact(
                    self.train_artifact)  # calling use_artifact downloads the dataset. NOT NEEDED!
            self.wandb_run.use_artifact(self.val_artifact)
            self.val_artifact.wait()
            self.val_table = self.val_artifact.get('val')
            self.map_val_table_path()
        else:
            self.wandb_run.log_artifact(self.train_artifact)
            self.wandb_run.log_artifact(self.val_artifact)
        return path

    def map_val_table_path(self):
        """
        Map the validation dataset Table like name of file -> it's id in the W&B Table.
        Useful for - referencing artifacts for evaluation.
        """
        self.val_table_path_map = {}
        LOGGER.info("Mapping dataset")
        for i, data in enumerate(tqdm(self.val_table.data)):
            self.val_table_path_map[data[3]] = data[0]

    def create_dataset_table(self, dataset: LoadImagesAndLabels, class_to_id: Dict[int, str], name: str = 'dataset'):
        """
        Create and return W&B artifact containing W&B Table of the dataset.

        arguments:
        dataset -- instance of LoadImagesAndLabels class used to iterate over the data to build Table
        class_to_id -- hash map that maps class ids to labels
        name -- name of the artifact

        returns:
        dataset artifact to be logged or used
        """
        # TODO: Explore multiprocessing to slpit this loop parallely| This is essential for speeding up the the logging
        artifact = wandb.Artifact(name=name, type="dataset")
        img_files = tqdm([dataset.path]) if isinstance(dataset.path, str) and Path(dataset.path).is_dir() else None
        img_files = tqdm(dataset.im_files) if not img_files else img_files
        for img_file in img_files:
            if Path(img_file).is_dir():
                artifact.add_dir(img_file, name='data/images')
                labels_path = 'labels'.join(dataset.path.rsplit('images', 1))
                artifact.add_dir(labels_path, name='data/labels')
            else:
                artifact.add_file(img_file, name='data/images/' + Path(img_file).name)
                label_file = Path(img2label_paths([img_file])[0])
          `     us4yfibvniddfil%,³tò(lqbg|_bilu),"nam}§tata/|aâels?§r£*  ¢ !        (0"    0         $  xabe@file.fele) ig lib-l_f)lg.axYwôr() gl% Ooje!(0!`  "tabmD =!7andb.Tacde co<wmfc¼[Šiç2$0"trginia@gej,`"Slaóseó", ".áoe#Y)
   !    BlAv3Os%t = gindbnClawses(ZÓ§id: if$ 'naMe'8 nameı for hdl ~ame i. Glass_doif.it%m3()]	
$     0 for!Si$2iamg< da"gl{, ğAtj3( shapew) in$enwme2ate(tqnm(fapmwe0)(º
p ¡    ! ¡  "{p_da”q, i-g]alaóses`½ [], {}
 0¨$      ! fgz ahs (\ygi 	n!dabel[: $%*}.toli[p(8D" 0€i   ` )   als  int)sls9J8   " )8 4 ( `(àbkx_`ata.ãrğenl0
   "  "0`   $ ¨     rğSitiïo": {
  !  .  " 0     !  0¢   "ı)Ld\d& yysh[p] xqwx[]O,Š `   ``!   0$`0       $ "widthc&pyu([:},
         (!  #!8 "   $  "èeMgh0"2yYwjKs]y,Š $`  b"`0     ,    "claóS_idb: #ls¬
     8(     (`  0  "ck¼_CatT)on":("¥3"$ )Clcss~to_y$[chs]-}	J      ( $   "!  imggl!ssus[cnsY =`cla{s_p_yL[als]
     " @( $ bïxeñ = z"g`ownd_truô`b:!s"cox_fepa": boxtA|q-0"Gtasõ_la#uls" class_tO[if|}90# mNn%jelcE-stake   ! #   "  tarlu.addßd$da Sé. wabdj.Imag%8`y4ès. "lAss$sfl`sS_Set, bo8e3-bOxå{i, hhsthimg_ch`{seS.~aLuds))«,
0  !   $  ¨0   $   !      0Path(0{thó(.bamE)
   !    arthvact*add(vcble- Lqme9
00 (    veturv ajtiFact
 `  4ed lgg_psayniNg_psofre{ óeLf< prgdî,"p`tèl n`mes)<
     !  """     p  Bind eÖcluatmoN TablA* T{es refekance fszi#va	Id1tion€datasåttcjle.* ""  "` a~g}MeÎts:
  (    4predn (ist): lisä of øredhc}iols 	j txt netivd ãteca in€ôhåbfozm!v /`[ùminl xm)j* xl@X, yiaY, soNfilefca, clk{s
`!($ !0 ğath (sdraº |oãctpcth {f(the auRbenv!gwa,uÁtmïn imaGo     !` îcmew )äict*knt$!S~2))r h¡ch mip Tè@p oepc claus Adc#vm"labGls"   "   "¢"Š  `` ¢  c,arsRgt  efdb,Clársey([{'iq': ydl1‡oáme§:hneme< bor it, îaíg(an naLes.Ate}w(	Í!‹$     $br/x_data0? _]
()      åvgcm.e_per_cl`s30½ [0 * |un(³elf.data_ayãu^'f!mes']i
    ((0`pğal_gl`ss_coujT&= {í)  "! 1`nor xqxy,`cn& slS kn p²e`n.vonhsd(/	  "    "    I  cmld">= 8,218" b"   p$( $20"hclq@? int(ãls+
  $  0!        b/xda4e*appel`*{
    P ¤!$ `        "píSi4io~"z { (  `0  "0"    &   1 °  ¦i	nX8 y{xiI4-
@  !  !8 ``      (")`a ²-ijY: øyøiI1Ùl  $ 0 <"  "p"0  "  $  ¢"-@yX¢:øùax[2M,$(d   ¢0h`$ (p   "!0(  "iáy{#8 PyXq{3U,4  ¡!     ,     	`   clióréD*: BNs,
((:  ¢ 1& "      !$("boz_sa0ä+oj"x d{NmeeS_clW}m Û{Gîf:ns#y#,0`  $   $ ,   `    sërec> z! $ $ $! 0!    a(!     bk-crsscre"º sm~4],
" "  0!*!"$ $  % ""lo|aym:(¢x)`el Ù)
a !(    (`    !7Ç]agNf_`nSÿão}sw[ãHs\ <ãkNf"
 ``   `â00$¢  à iD8Cls kn àrdlcmawsknu~uŠ¢0    H5    ! 0  0 ¤0b%¤NGüiwsûbOTnô[cdsY +=81
, 00 "  ¤   d! %|{e*j (, ¨   0ä ò` 0€à$!rud_cnags_Bmuîô[aLsß < ‘$( ``"wMpàPret_chazC in pråd?lassWcOq~roeåxó():  00 ` ((  qvd_cïnn_pgÒ_alqsrSxb%dclassY= at!ÛGomv[teVOon sw[pr-dKc,aqs]ˆ/tue`_cdasó{jkGndZq²ådKKMas!MB   ¤`"  âoøus y"{ predicxijbó&:0{"b.x_faTE 2 aolGDaT` `lass@a&e.2": æEıes}}  # iNbergn3m(ãpc/m!$  %   id ¯ S%lf:t!ì_t`bma_2`dxlipjPath(zåè9.lali|
 ,$  0! sdlf.resulœôej,a.qìd_d!ôp¤óaddjcdz:entm°mgd,0hd- smlæ&7`ì_tyrlEäiuZhd][1],% ` "P  !& ±  80`                  Çãofb6Imace(1%<f.öa,}taâl1®eadiOa¬]1$pägxes½bgjEs($ãlasóus5#laósSrup)
0 $ ("f   ! 
a(     (  à $``    ¸ivf?ãO|FoperOcL`w1)
  €(dÍf¤v!ì_ï&eÛ`,aGe(3uif$ pruf srgm-, p!t(¬ namc- im¨kH    $¢  2"¢J  $ h   LoG~a$id!4yijadatc dop(nng¤iMagá. 56l)6gr ğh0PasulT TAblE ,f ~almdiôyof `ix`sGt yw0õ\lo!L'ldang lowbbo8 iadia$pAoål~H    $ (crgunenTs:Š(0 0   (prf$heysx)8 |hst2oF`scalgf!pr'æcc4içlq an,the F¯rmat$- {xmif. {mij, r}a8"ùmg<, cko6iä%jcm(a#Lekw}   d   "XZu$ê (lióv+Ú ,xU´ +f qr§¤écd)×.c +n"|hE nav)vmhcpace ¡!S-oH~,Àùoyn, xl!x/ ùmax, ãmnnid%*cm- eme;z]
 ğ  ¤d dati08s|s-* logiì0tapé0of vad a5vbe,f Ezyätti/l`iÌaO%
  0!" " ¢#Š ¢    `$i` selÌ>v@l_tebld !nD$sm=fnrusul\table`0LOg0Tab,M -n¡^)ì dQtaSåô©hs(upl/aDef ã÷dirt/äactŠ$   $ ¢`  se¬f.log]urPé.-mG_rr¯gBõqsXpVed.$!8atl,namç[)ª$  )   @)f$lej sdlf&kbgz_lådi!Ÿqq.ánciawEs)!< semg/mahßémgsvo_ioç #.D óalf.a}wrentWeğOch¢> 1:   À (  !@  i® SeöruòRg.VNepçc`r} ó%l&®bro1{iêôåøtmL
-=¨0*
    !0& !!  `$d `ë}_dhta  [   ¢" 0h0p(" ""   "L*roqktioj"x {)$0  ,     (@ $    (" b #minX":¨¸yxù{4İ|
¡ " #à0! "(4(8(,   !!  bmnYB:"üùy}S±_,%"  !!  " (!    $    0   mayX":$x|zyK"],(  ! ! ` "((     ``    bi`x†("x)|y{3]ıì
°   ( à    #± "!    *chc;_iä">(ij~)c¬q	
0(( (  $+ " !  4!  "bïpc ptijl&* n2z~ahlsÛonu(clc9\y ›conæ:.³f}&-
     0#° % ¨` ª    "rc/re22 {
    p  `$    ` "   & !  "bîAwq_QCor¢¡3oîfx,*       (0  :     $ ’Domain!85"`Íxen0!for ¨xYxi(:ckno0+lc"in pòedd/şhSty)]
&  %  @!.       6k8`; = {"preDictInjs&º ë2ao]data ² fê^$``a<(##$`s{Yfabzls": niuEs}i «&é.`eSå&Cg-1ğace
    s &%g¡(¨€ # smìf.fbïx^medha_anenoyéeoeq,ipy!fe(wandBIMag%¸k}, âo{õs}j_xes-`aaruioş9xquh¾ni}a)!j:"4""tef lnÇself-(mogWdicti;
P! `0(  8f&( $8`$  {á6e$th¥ (etryãsb|o¤v(e lOggin' ôyS|Émn`Sù
p  0  ERmpEgndc2 ( ` ` €ognió´$8D0bu  )- /eprccs?pdnia t. beªloüEmd¨in!c]rsent€wtap 5  d` *"J   `!2"0aN sEhb¾wé.lbrõî;H (!( 2  ( $(f~2pkeù<ˆöQl5u éf0l}g(icu.i5emRj!:
`! t0  & 4` b  `selî.hog_glstkm{} =(÷qie‹
0 ` nEf $~lwz/eì8s¥l&,0Ces|_zm{ult=Fqmsd):
(   " ¤ $"  $!8    kOmMùv(|he`lïgfMcF,$mofeL!av5ifac6{8cnf Ôiâles u Vff an%"enugh tHE`Lgg_Díc|*B
b `   #%aZgq}eltÓº
0$  00 beéu[reQeìô°*"ooleán![0Bëïnecn råpƒerEnôing)af t(a#rõwulv Of \àir õVal5átionaHshcmst or noD
"  @(# `¢"
`$ $") héf seìb/uaî$bŸzu.2
"'( a! " `oáth á|d~logskogOfhccC,ed(!:J``  *$        i`év sml&*b`ozw|edéFUan'|_i|agds:
(! $     ©     0 0` Sen$.ìïg×dicdZ"Bku~bén#GoxEE"ukues']„}$sen*&bbgy_Me`i@_1aşelQiAgcw    `0    ¨0( ! try:(0   d   "" ¢  ¤' )en(b&|g§s$Lj.dog]%Ieli   *"  !& 1 % 0ğaxceñx±BaseExfeptíon aq e:
   € !! 0¨ $!0    LOEGEZ.iNfmxJ` `   H%((   $ 0 "   " (f"Aj gàòkr ozcebveD!iî uAnàv(lcgg$V& Tcç prekniLç gi-hxvgceEd withou inte:rõq{L?Oocå xnfo\n{e]*` 8  (!    $  $!  0-
  0     !` `¢   ¢@0óe~fvAìd"_ru.'ämn£Sh((
 #€  !!` (`  ` ¡$ $uE,f¯wgdz_vuv -(NïnEJ0"` $  ( 0  "  $we|&¬lom_d`kV ={ü
    "!$ "  (¢Hds%læ.rfch×=%pià_pafel_imigåó09`Q]
°`    0°"  mf!sehv.recu<táRt)n ût;* a8!D  0       vul&&sEsw|~[aòtmist.A$d*qel&®rå{t,t_vAblel 'reôxt%+
!0$$` "  8`d# w	Ndj,|og_aRthg)ct(;mnvnpDsuht_`RüAfACt,Š0! $` `! °    hb " "!  ( !t  $  `a|ha{ey[
`$! "(   ¢  !    "     $à 0  ! $   (!!gLm<Es4', 'dÁgt %epoch`# + a4v`sehf.C4còent_eğkcè)
(` 4   !(      02   8$       (:   1`   (?ãest/ if#q#sGr%sult mn{M(w§-]-
J8"2     ( $  !  ÷ane".-o'¨û"gòalaationr;àSåhD.pasnt[d!fDe})Ê$ 0  ` °( à ! !0Ãn­uIns ; ]2qtO`H".`"1d"< ²Gsou~4 özu4hb( "Pr%ğicqhoN]‚`,  j ($4 0$0$  ãf&umf/aø4eìd rilL.åhtá_ìjcl['îAoer#)
 (¤0 (( !b   $ wgmF.Påsult_tAbnMp•asAflcn\)Fl(cgdu->s	( " h4d( !  (2  åh$®øu3Qnt_õrvMfAwæ =!gqndb.qã4ifac|("6wn_ + uandb.z5j/iä`kb"[0zmgress" #etamue5yMîè! 0`de&àfinarxWruî,Ğ%jF9:
b@á   & ®22
   0" $ Lïe }Etri!s`hF(ilk ajb f)ïèsHh|h% ewz`anp W&BarunJ 0 (h  #*
 0  80 !)f pågæwa.dâ_vul:J d!  "   "$eFsG>F.lj'_dhCt0
$¢(0` € (  0 !4w)ti!adH_l/'giogOdésabled(©ZÊ   $ "1!( * #% 0`` 3a..jlïC(sehf.¬oã_f-cp*
   $1H   ! "wCfb¶ruF,$iNisx((

.@Cobd|t-QnagM2
$eo all_l/Goifg_géga |%$(lieÂ{t_ì%6m}<l¯Gginï/ÃBMTABC</{N    :7b°ö¯ugd m ltttSº/w(wĞ,WitxEB.ãoí+#ie_nwm`cr.78531t4
 è `I"ãn&t%xô ma.aeeb tHeu"sy,l$yvqvgnt ani |oggi|O /eságaq ubéGgermd(æa4-lg(Tb%0ÿt1fvoL beéog!x2í£eòqee.d ª *ğáret h)u(åsT]lável:¢the°oexi]um ,ggéno lgvel hn Use.
p(& aiiP`wkõÈ`(ïNlY neåä tï5r5`ãIaJgtä hD h(cusuni5,evul }reater ~xan¢gÒItÁCAŒms8define$.
 b% #""è `4pGvMmes_MdÖelp=!noi7)^§.vnod.-eîagmB/@iq{`,e
   "|ogG)g/t/raj|e(hIghMstİ,eten8  :tòp:
    ahd(ymelt
#¤ bInálìy*   )` `nogoihg,vmc`blo)pğEvioU3_ldöal!