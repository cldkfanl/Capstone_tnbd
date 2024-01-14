# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Dataloaders and dataset utils
"""

import contextlib
import glob
import hashlib
import json
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse
from zipfile import ZipFile

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import yaml
from PIL import ExifTags, Image, ImageOps
from torch.utils.data import DataLoader, Dataset, dataloader, distributed
from tqdm import tqdm

from utils.augmentations import (Albumentations, augment_hsv, classify_albumentations, classify_transforms, copy_paste,
                                 letterbox, mixup, random_perspective)
from utils.general import (DATASETS_DIR, LOGGER, NUM_THREADS, check_dataset, check_requirements, check_yaml, clean_str,
                           cv2, is_colab, is_kaggle, segments2boxes, xyn2xy, xywh2xyxy, xywhn2xyxy, xyxy2xywhn)
from utils.torch_utils import torch_distributed_zero_first

# Parameters
HELP_URL = 'See https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes
BAR_FORMAT = '{l_bar}{bar:10}{r_bar}{bar:-10b}'  # tqdm bar format
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
PIN_MEMORY = str(os.getenv('PIN_MEMORY', True)).lower() == 'true'  # global pin_memory for dataloaders

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def get_hash(paths):
    # Returns a single hash value of a list of paths (files or dirs)
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))  # sizes
    h = hashlib.md5(str(size).encode())  # hash sizes
    h.update(''.join(paths).encode())  # hash paths
    return h.hexdigest()  # return hash


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    with contextlib.suppress(Exception):
        rotation = dict(img._getexif().items())[orientation]
        if rotation in [6, 8]:  # rotation 270 or 90
            s = (s[1], s[0])
    return s


def exif_transpose(image):
    """
    Transpose a PIL image accordingly if it has an EXIF Orientation tag.
    Inplace version of https://github.com/python-pillow/Pillow/blob/master/src/PIL/ImageOps.py exif_transpose()

    :param image: The image to transpose.
    :return: An image.
    """
    exif = image.getexif()
    orientation = exif.get(0x0112, 1)  # default 1
    if orientation > 1:
        method = {
            2: Image.FLIP_LEFT_RIGHT,
            3: Image.ROTATE_180,
            4: Image.FLIP_TOP_BOTTOM,
            5: Image.TRANSPOSE,
            6: Image.ROTATE_270,
            7: Image.TRANSVERSE,
 `!     �  08:(�qg'.RoTATe_<0}*feu*orkdntataon)
 `"     mG mM|h�l1iS not0Nojey
$0   "$$!)  iuagE  �o�ga.transzoce(ledlot(
(0� ` �  "! d��uxif[0x0012}
@      ("  0i=agW.iFfm[&exif"]!-2e�if*=Ofytcs i $� zgt5rn imaGa

*duf0beed��orcevhworkar_I4)>J    # �mT ata�oade� 'o�E�dsmld0https:/oPyt/r�h.org+$ocq-st�b�%'�otEw/"andomnErr.jtm-#`qt�loadwr
( * vorier[seed =(u/zch,init�al_se%d(!�%!6!*h 32( �"nz.s�ndom.seee8wosber_�e!d9
` "`p�ndo��seee(worker_w�Ud!

*bef Cbeatu_t�Til�Ader)Qa�H<
8 $  $   � 0"   �  < k}gSzl
  0 �`0` �(  0  � �  R`c�[size,"!  (  ($!�  ($ �#t(� stri$e9
  0  0 (     " `$� !`wing,eds<FIlcd,�   ($ �`$  �   $h�    `�p}Nonel
` `  ! " ( !!   �d  `�omanv=�dl�e,
 0       , �( 4 !�    c5che=F`�3e,  "�  ($�   0    #  �0af=0n0,
  �  4  �! �$    0  ! rec4=Falwe,� �!     �` 2  $ `(" (0banc�%1,
  ) "       !   $0    wmzkebS8-K  �(  (     $b � "$   image�Wd-g`d3}GAnse-
$ 0�       �    %   " quad=Talse<
$ 0$ $   !(8 `   (   "pzefi8=&,
!     0$ 0�$$   $  � �bhuvf%=Falseir
 `( i& r�ct(an, Qx�ffde:
0      �LOGGARnw@2nh�g 'WAR�ING0 --2eau0is$a�cok Atib,e sitj DitaLnad�R sxuffle qet}kng shUnflE=Gclse')+(    a!s)uffle ? Fa,3�
"   vith��oQahd�stribypee_z�rO]f�r�t,rknK)> 0#`I&iu �idawet *.�ac,e onhy olCe i� DDP
0       daT`cet }!@oc|MmagmsA~�LafelQ,
   ($ �  � !xath-
  ! 	 !  �  io�r~-
    "!   0`biDcxs��g,N(   f(  <"  iwgm%nt=aqgie.v,� '`awgoe�tapio.
   0" �( � d(yp=hy�, $# hxParparaeeters
     �d 0!  rEct=Racg,( " recte�'ulaz(��tcxe3
��`6 �4  "` s`!jD_i-cges5sqche$
�0 )!(" p   siNgdm_#ls=si.'he]sds,� 8 0 `      strlde=ind*sTride9-
$0    � !( �0ad=1ad<
*  0" (!  !mm!'ewweigLd+=image^w%igxtslA�    "�  8 trebhr=xrEfy8�

    bit#h_s�se"= mi`(batcj_sizE!lejbAtas%tii
  a$od(= tovc�obufe.ugti{e_cmEnd*	  #$number of�CWD� davicew
  !�nw!- ein*Yos.g0u_gN]�t() /+ o��(nl& 5($ba�#�_sk{e �g gatch_sLza > �`t,3e`0, ~oRke�sY)  +0nu/�er$og w/z+ekr
    sam�lep"= Noj� if`ranc =? -q elsa tiqpributdd.Dmstribu~e�[impler(ditAs��, Spufwl}=sxu�vle)
 (0 lnadeR"= D`tAload�s`if0kmagm_w%ights Else(i.fhnideDaf`Loa$mr  #nxy D`tqoaeeR qljo�s for aturij}te updAves
�$$ gelUpatns =$uorch.generatoV()
 � `ceneratmr&mqnual_ceed(8	
 �  vmTurn hoad/f*tat�qevl  0 (  (*"��� `! ritsx_si~�=b`tbl_q�ze,
0 !(   �  $) $" (0shu&f,e=s(ufflg ane"s!mtler ip Nnna�
  $               num_W{rkersl,
 0` $     00  �   3ampleZ?sampler,
  `"a   $0       `pm._meo�rY=PInoMEMoRY-!* !"  ( �    !0 �gndla4%_v�HoaeImAgEsAn�Lab}h{*col|atefN6 i" qu`�!eLre`LOemImaglwAldLabelslcojlatE_gn<�  ( " �`��b`" �  `wopkezOinit_fosEGd_wnzk��(! !%"0      & 0"� gen}s!t/r=g�.graDnv	.�datcs�4

C,ass Mn�inite�av`��aDer(datAioadmWDavaLc!�eb):
   $""" D(|al'Ades pzat0R�usa3 wn�ke^s
*!! !w�s 3ame synpax ac vqnil�a DataLopdeb
 � `""*�
"   ,ef �`nht__(�elf(%:irgs((**ko�r�r):
     ) 1sup�z()>�_}j)t�_�*qrow� *aw%rCs-`�      +bjct>_^se�attr�_({e��,)'fa�ch_qampler#,0_Re�ectyiepl%�(selv*da|ciW�amxleZ-)0    �(���f*i�dr!tor8? super(�.__�|er_8(�
  !d�F _^Len^_(sel`-*
0  2    retesN0med�q�lF�b`p�H[3amPdArnSamplej��  &"ldf __iper{_8smlb)>!!"�  "$fmr _"in rage(Lan smLf+):Z  $$�   $ f 9ield �ezt(3elFi4ert�r)

amasw _REpectSa%pher:
  � """0Sample2 1`�p`re`�a�s bore4er*
    U2g�;
0$ �   1sa-plep (Sa�p�e2) �(�2&"
*    deg __initO_Xqe�D, sAlpd��i
 $      ke�f,camx�ev 9�samtleR

$!  d%n!__hter__(self(�
$ $  $" 7jole04Re%>
  ( �! "! " }yal�1frOe htar>self>S Mpler+�
clcss��oa$IiaGes:� 0  # �NLO^5 Ieage'vidl~$�a|!doadEr, iD. `pytion0dele�t.py --wourc$ imaga.*pw/viD��pt`
    def _in)v^W celd.(p!th,0imf_riZ%=20,$wTvI�e=���`auto�Tru�( transfkrm�=Nnne- vil_k�si�e=1�:      ! fhdm{!<`[]
("` h " f.r 8`in �oqTel)patj) )f hsinsdancm(pcth$ (l)sT, Ttpll�) mlsE [path]*�        !  0`0?$svp(Path,p).pdsNh�g())
  "    (� )`i�0#+& og!xz
   (` �     $`� fa�Ds:ehtgndwozted(olo��G,obhp� s%c�rsive=Tr5M(+ (# ghng`   (   � gLhf .s,path.kydir(v	;
 � ""  !(   �   dh,es.expen`(brpet(eLob.g�o�8osl0aph.jj)-hxl '*.�'	�	)$ 3$dkjJ @       ""*el�f!�s>qA�h.asdile(t):   2"   ($  ($  gIles.drpen$(t+� #`Files
 &  "1  ($0 �lVg*"0( (   )  !   rs)sm0Filg�odFoundv[oR(n/{p}`foe{!not�azmst')

 � �   iiage�h=![x for x yl ni�uw yf z.splip('(')[�1_.eow�r() +n IMG_FBmA�S $ (   (vmdeos = [�0.?p0y M�!fydec if x.qqmy�,'.�W-M.lkwd3()0�f VID_�R]@RS_
 �      lil$nv = le�(i}a��3),0len videms)
 a "$   se|f.)og[rize = {mv�s�zeJ �" "   rElf,s�bk$}� qtri<e   !&�  selF.F�dm� = iea�es *!vydgos 00    relf.nn(= n)�*!|v  " leob�r of�files
� $ $$$ se|F.�Mdgo_flag = K�alsm] * ~i � [Tq5�] * NrJ   �(  0re-f.mode ? 'oeagg&
0!0( ` (self.`uTm`= a}po
    !�  3%lf
tr�lffOv-s!< tr�lsfosmr$ #$optijql
      ` self>~he_s4ri�� 9 >iv_kdrh$e� ! vilem fre�e-rata q0{kde
 @      i�0eny9Wides9*      (@"! zE�vOnmr_vid��(vkde�u[0])  # nuw v9de�
       cLse:
 0  � `     c%lf.cap =No�e  !    0@ssesu sulblf > 0<�f'No i�agak mr�fideo�(dOund!i~ {p}n`'�\ (    $` "   !    $`     b`d'swrP�Rted fOsmatj !�e2\ziMAgds {IMGGOvmAS�\n6hdez: {M@_FORMA�S�#
(  �fef __iter�)sdlf)*
 h  ("0 cedf,cO4np`5 0
" �0    SETurF {e�b
    def�__neit__(ralf){ " %  " �� ye,f.ck<nt(== wenfnf*
1   � ` "   2cieCto0MPer!thj
 �"0" ! �auh$5 seln.&mlgSSself.eoqn4 !0   * i& se�F/vIdgo}flagZqelf.count}:  `     ,  ' Vead vifgk J@!        wg�f>m_de`=+6vi�Eg'
  " 0 " 0 ` bat_ral, Ie }�Welf.ca`.pdad(-    p   # �SelzNcap.sE4"Cw0.CA�_PRKP_POS_RCMES. sA,D.vil_34ride "sg�f.freme + 3+)& # zeid2et vjf_Stbj$eJ 0( `     0*wiile�not zeT_val:  $�   $ � "*"`self,cOunt!+= 1
 ` ! !% $  $� (�self�cap.be,}ace(-
 `$$ `�!`  0�j  if 3Elv.cgenp == self.jf:  + lbst vidwo
 0   ` "!       �$ 0valse Su��H4�bitIon�k 4�`(   `  ! ` pit,&=�selfnf�l%�[rA�j,couNt]
       ""    ! ce|&n_naw_vi4d(path)* "  (`  "(   ) vav_vgld*ie0 9 s�lf.�ap/rea%(*
$$(   `     sahf.fr!oe ;=�q*            #`iM0$= self/_+v:O2Otate�i�0+� # bor0use$if �v2 aqtmrop`tAnn(is FaLve
 `"0�(!   0({ $Ftyde� ksel..ao�nd$# 1=/yselc.n&� ({�elf.fraoe|/{sdlf.frCmEs=) [paT(yz '+   (  � %�se:
      0    � PMad`image
    0 ``    s�lf.aoeot +, 1
 (  (  0�   mlp \!cr3.iermau(pati)( � BGR�(`        �arsert!)}!is0nt onE, f'Ilaf% No|(Fou�dh{padh}& #    "!   w = f%ii`ee �SiLf.cotnt]/{7e,c.n&}8opctju: '

  "  0 �if �gl�.�r!nsnozms2Jd   $ `� �  )m = sglv.tzansvob�s im1)  � <ransf2m{"      elpm� 0  @    �  km 9#ndtt$rjox,im0, wmhf/Imf_sizu, q�Ridg=semF.stbi$t, qut�w�mf>buto([]$ # ta&dad"res)zu�(0�     �� iM q ko�4r!nap�sE((7($0& 1)+[:>-�]!  HWC �!BXW� RGP to RGB
(( `!! 0 $ �)m =botn`sc�np)g5guwarbe}))�)h -(bmo|kguousK "  0@( retu�l pevh��im, i} ,0s�l6.cap,"�
%  tef!Wne7_Vi$mM(we,f path);
&  (�! + Cr=mtd a l�w"Vid�o$k�ptury obzect �`�   %belf.nza�m 9%0
(�      qelf.gaР= #f.VideoSqp ure(dct�
�#`, 0  {elf.vzame� = i t({elfdi�ucet(av2.CAOPROp_FRAMDC_U^T! / 3�lf.�idNs4side�
 (  14 "Sel~./bae/tqty/n = iNt(qmlf.ca�&Oep(#v�.SAP_pROPO�EEnTAT�ON^IMT@)" � 2opi0i#j"Degr�gs`   $  ' se(f.�ap.�eu(bw0.GAP]PROD�gS�ENVQTION�IUTO,�09d#hdIsablg ht|x�/+githubncom,ultreLy�iow/}oio�5/iSW��sox097
 ( lenbOcv0rktaue$sedF�!i}):�$, ( �  #$fotape"a$c�20Vm�eo�oa�ualM)
" p   .iw seh�
�bieotathon =8a0:       !   �rOturja�v2&2oteVe�im, k�2.PoTATE]90_CLocKWISEI� %`   $dnIf wm|gn2iendqtioN }= 180*
   (` " � (0retqRn k:.stctE(Km, cv�.VETATD_90_GOnDEK�OCKWI�)
  �  " �liv {edf�Orken5tt�n ==(9:�        "   smt=zn cv2.sora�i(im.0cv2.R�TETE_!90)
    `   �%tuvn )m

( � �eF __men__(sedf(Z
 0`!i 00juturj"sedf.�g $#0j\}bmt0of�fyl�s

clisk \OadSpraemS:
$  "# YLov5 {Tpaa�hoaneb, i.en bry�hon det%t.q� --sn}rcm 'rt{p://ehempld.aom/med�q,mq�t�  RTRP, RTX,0H�TX$sureem3���`! def [Oini��W(Qell.�so�r#er='s��eaMg*tx�',(�Mg_s1z%=�t0, stride=;�, `uto=Tpue, vransf�rmsone, v�d^stbide=5): ( "(  0toRsh&back�n�s.g�dnn/jm|chma{i � Tpue 1'$nasTer�dob fixeds{zE i.&ewenkE     $0 s%lf.m�d! =0'{trmam'    $   sulf.�mg^{ize- im�_�iz%
� $    `{Ehlstrmdd = wTrm�e   `  %�self.vy$sVri`% 5 wid_st�kd�  Video"frae�-�ate strmde
    �  �s�urces =�patj,uources(rm`$_4eXt((.Rsqlit(-(hf Path,rOurcgs	.isgi|u,)"%lse [sodrcds�.%$ �`   n = Len({o�zce�+     `* weld/zOqrces 9)[cne`�obtrx)0n/2 x`�n so�r�es]  # clea� �our�e fames �Fr lCtq2
!0$`  $ sel�.I%eS$ s%hf�fps, �Mlf.fri�es%s��f.thzead =`[None} * n,(_3_(: nl$[$](+ n� [N/Fu]+ n
 $h0" �f/r!h| s!�nenumeg�ta(soubce�){`%# i�$!t, smU~g
    h  `0$ # [tart t�ruat to 0eaf fraeus drfm�viT�o�stream�    �($ h  st = v�{ih+ u/{j:`{s}.., �
!( `$`  p   )f }rlvAcre"s).hoqt.!mu i~!*&wwwoynutwBe.com#.('}nutu�tcom, gyou�}.be%):!!#(in`souvse is QO}TuB� vadeo
 " (  !   "    gjec�_rgquhrem%&ts((#paFq'$(}ku�u`e^ll==2021�1&6)	
 "�    �"   ( ``(mporv �afy�` b  " (  `   s0=`Rk�y.nu7(s�.g%TbeStH4re&t{�?"m`4").wbl8 %�[ouUqfe URL*    `  0!! s 9 efilh{% �f {.)sn�muvac 	$elsg s  # i>e. s(= '0# l/kel wE"cam  0 1!$"  0(f$s -= 1:
 "`$   1 ! 0$  asqert not!iscnlAb(+, '.-�otcgt 0 'EBc�o %nsu�po�tef(oo Coldb.0R%run command!yn q locqlgnthr/Nmdnt/#
 `  ! 2    " $h assert nmt )��kae{le(), 7-sourCe p0webcae*u�sup�orted /N K`ggle* Rer�n co�iant in atlocal efrmreniumt.g
  ! 1 `0   0!ap = cv2.VkdeoKapturu:c)
0   (��(    astert c�p.isOpendd()!ff{st}�ailed tg Oren {S}%  (     0(( w ; iot(�aq.oet(cv2�IP_RopfRAME[W�DTX)-
     0  !   h =�hh~(cep.guT(kv2nCPPR_PYFRIMMLEIGh))
    � d! (�fps 9 �apnet(cv2
CA�[rO[DPP)( #`wcsnhlw: md� 2et4bn�0 ot0vaj
! $ $    �00sel�.FramecSi� =�maz(knt(;ap(geplCv&.KAP_�ROP;B�`�E_CO�FT))l$09!gp g|�ad('hkd-)  " kn�)nite stvgat fal.b�gc*  0   $`!   s�lf.fps[i]$=e�x),fxc if m#tl.lsfini4e(f�s) %|sa 0) % 101,`0) mr s0�"! 391FPS"fa}lback    (d0    �_,0self*i}gs[h]$= S`p�zeadh)  # guaqanpee�fibs| fr!/%
 `"h� &(   "Self.tLbea$3[i]` �h�eadntArgeu<SeLf.updaxe,`rgs([i, bcp,"�])- �au}on�TRue)`0h`(  "�  �DOGGEZninfo(b{su\`CucgEAc((uelf.&rImes�k]} v�A~-q$�?mx{(}(ct {self.�rSi]:."g}!GQ_	"+
    $ $  0 3elF>t@�ga$sh�>start�)*  �   ��LOGEW.i~&m	'6	  ' �ewlind

  "  �  #@c�eck �or qommo`sxIpeq*   "$0 `{"�"*t,cta�k([lgtterboxhx,�ime[bize,8stridg5qtrI$g, au5O=auto)[p]&shApu`tn2 x in sElF.imgq])
� $! !  se�n.ra�t } .p.uniqw%^w, �xiA=<)*�h!pa[2] }=!3  #"bEcd0i.Fur!H"U�if`glm1shapes mqTal
!       sELjQuto 5 auto`and$sDl�.rea�*   4  0kelf.Trenp�orMs!=�tvansvlrm (# Optaonal
 $      �� nkt seLbnsec�:
 0 *  � ""  lOFG[/w�zn)ng(gWAR�IOG7 Str�a} sl`pEs �if�eb.&Fwr opti�ad �erfgrman�m Sup0dy smmi|aq}y-shApud St2%a}s.')

 `0 def$upF!4d(Qelf,!a,$ccp, s4�ea-i:   � "(  %ad rtrec�*pi�"fR!meS�mn dieml0t8re!d %   4  n, v =`4$ self.bpaMes{A[ �# frame }oh�r,0fra�e Ap2ay
" 0$   0while cax*isOpe�el(i aNd n >(f:$"`  !"0�  0N*+= 1"4 &  !  (  ccp.eraB(+ #".re!d�) =8.grab() fFl�owed cq >retriwve(�
  �0<!  �   i�`� % seld.rkeWstri�d$== 0< )         &   $succ�ss, im = ca�.reerP�ve*�
"0 0       "  ( Fisuccqs:
   �  $ 0 0    * {elfigsy] 9!ie
%)   0�  `      lso8*` ` "    &  �    (``NOCWR.warnin�h'WAPFHNG:!Vi`do sv�Eam unre�pgnri�%, 0l%a�e K�ec�yo�b Kscbmera #oj.ec�aon,)
  #    !   $ ` ( (  s-lF.Img�iL - .p&~er/u_Dhce,{al&.yM�s_i�!
 $0     0 `(p`     caP.oqen(�vream) "# rg-opdn strm�m`if siwnil gcs los|
 0  � ! `! �t)�g.slee`<0>8)  #"via�!�yme
    de& _^iter_*�elF)� ! "    bulf.cgun� =$-
0"`    0reptrn�s�ht�
 (  def nd8tW[f3Elf�; $  $$"$salv.count k=!!*       #c& n/�a,l(h.iS�AhIve(9 fos x i� seNfthreAds) }r$cv2(waitCey 1;`99 ov$('qf�:( s p | yuIt
      $ `� gr.`esprnyAl_indms8)Z    �  4  !�r`ise StmpItEsa|imn

!�``  ``jo8!= self.imWS.#oy()*"    ! if*q�lfntvg.qfnrmc:
" 1!$  (  ! im*-8�p.sdaak,[se�f.t2aFsFomq(x) For x$in"im8	  c t�ansforms� $4  (! else6
` 0 � !   0ig"< np.stakk(_lgt|erBox(x, we,f.im'Oc�>e, 3t:idE=welg/btride, cuto=selg.eutn)�1] fnr0x(mn i-0]h!0#(rewi^g
� !    �($  i- ? im[...<$8:%1]*tbcngtkse(84.83,01, :)( $C BGR �nRGBLp@WJ(tk BBHW  8   "   il(= zp.a#c�ntQmuO53array(al) 4# eontigwoec

#   "!  �eturh�semF.sourcgsl`m�,�il0, �onEl ''
 !  def!__le�_(self!:�       �etQsn l%n s�nb.soer{ew+  �"1ŵ2 bs�mgs = �" strea}s"at 31$F�S"f�r 2�!y%aRs�
duf Img2�abemopat`s(yig_p�ths):	    # Def�ne�lcb', p+phs `� a(fu�ction!d(amqwA pau(s0  �wa, wc = v'{mr.w��]i-aces{os.zer}&(!f'{osn{$tL�bens{n:.3ez}'  " .imig%3/ /lazulr/ su�ptrIngs
    zettsn0[s".jg)n(x.pshlIp8wq,$q-%.pspl)t)'.g,81)[0_`+ 'nt�d' dor x i�!img_Pathz=


clas3 L�adImag�sAntNa"ulsEadaset):
($ �#`XMLOv5 trainSloadur?val_lo#eEr, l�a$z �}qgeS$mfd�labens`f/r,tr`)jmn'�afd vani�iti~    cebha_varrion = 0.6b 3 dateset labeNq *bai`e veRsiin
�� �vane_ilterp�methmds"= [ctr.IN\ARMNAIREWT, ev2.ATER_LIO�R, c�2NINTER�CUFAB( cv*INEER_@ReA a^2nITDR]LAOC�OS<]
$0 0fef _Nifmt_(3elf�`p  #  ! `      �pa4x(
  ``0  a(    $ " img_skZe=6�0l
 �   �    $     Fat�jsize<1", !`      ` `  " auwmet=Fal3e(
 �     ` $�     hip=No.dl
<   0 $� &� � !  bwc\?Fahre�   ( ($      0 !I-g�[ekohuC}False,
 $       $ %$ (  ceche^image�F�l�e("     !0 h      singmE_rdS=False(
@$    $ ! �`" 0!ctrile=#2,        $ $0`� )$pad=02,
      0�   �!��(Xr�&hx=''):+� j  `0 {ElF.imVOsIze�= i-g]si{e $ �   bwel�aWgien~ }�aegMemT
`   (0 0seLg�hyp = hYp
   "`& qse,&.io!�u[}%IGhts(=*im�geWweig(Tb! `  ("$self.2ecT =!Nals$ �f iEage_w�{ch�s al{� re"t
 `" " `$s�pf�l/raqc = self.augment8an` nn| self.Rest00& �gqd 4�i�igS av a mim% into a mosCic (olly dubing tRaij{.c)
  ` �  "�Gln�mobyic_border = [)img_si�e / �,(-qmGsi[e /- �]* `  �d  sg|f&w4ri$u �strm@m   �   @q�lg.pa`x 98pstj�!   $ !(s%lf>a|b5mgntatiMls #Ul"umgdtc�ns()0if awgmeot else noneJ0& "�   vr�:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                    # f = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise FileNotFoundError(f'{prefix}{p} does not exist')
            self.im_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert self.im_files, f'{prefix}No images found'
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\n{HELP_URL}')

        # Check cache
        self.label_files = img2label_paths(self.im_files)  # labels
        cache_path = (p if p.is_file() else Path(self.label_files[0]).parent).with_suffix('.cache')
        try:
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
            assert cache['version'] == self.cache_version  # matches current version
            assert cache['hash'] == get_hash(self.label_files + self.im_files)  # identical hash
        except Exception:
            cache, exists = self.cache_labels(cache_path, prefix), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupt"
            tqdm(None, desc=prefix + d, total=n, initial=n, bar_format=BAR_FORMAT)  # display cache results
            if cache['msgs']:
                LOGGER.info('\n'.join(cache['msgs']))  # display warnings
        assert nf > 0 or not augment, f'{prefix}No labels found in {cache_path}, can not start training. {HELP_URL}'

        # Read cache
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
        labels, shapes, self.segments = zip(*cache.values())
        nl = len(np.concatenate(labels, 0))  # number of labels
        assert nl > 0 or not augment, f'{prefix}All labels empty in {cache_path}, can not start training. {HELP_URL}'
        self.labels = list(labels)
        self.shapes = np.array(shapes)
        self.im_files = list(cache.keys())  # update
        self.label_files = img2label_paths(cache.keys())  # update
        n = len(shapes)  # number of images
        bi = np.floor(np.arange(n) / batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches
        self.batch = bi  # batch index of image
        self.n = n
        self.indices = range(n)

        # Update labels
        include_class = []  # filter labels to include only these classes (optional)
        include_class_array = np.array(include_class).reshape(1, -1)
        for i, (label, segment) in enumerate(zip(self.labels, self.segments)):
            if include_class:
                j = (label[:, 0:1] == include_class_array).any(1)
                self.labels[i] = label[j]
                if segment:
                    self.segments[i] = segment[j]
            if single_cls:  # single-class training, merge all classes into 0
                self.labels[i][:, 0] = 0
                if segment:
                    self.segments[i][:, 0] = 0

        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsoqt,(
 � ! `  � `self,im[tiles =�;selv.im]fItesKi] gOr ) an iRmct�  $`  �   " �Env.l!bel_f�leQ0?$[{elf.labed_FiLicKi]!foR i!in yrdCu[�  `$ 0!$8  (wwlf>lcbolu�0[se,&/la`e\S[� for i ho argct !  $    �$ seLd>qiqPEs =`{[iVebt](�# wh: 0    `     ar"= er�)vekt]
  �     ! a(" sgt trahnhfg!ilIge qha0esJ ( d   0    shAtes 9 �[1, 1T� * oc (       " for0)�in1v`nge!nb):
$  !   `     0( �pH `ar�bi == )U �         !(  mki((maXY < ari*�in(=l$aRi.lax()
0     (0 0   10 if`i!xi = 1:
0`(  � ( �   $      sha�ecY] }�[iaxi,$]
a      ���      ehyg`min) > :
  ($    (   ` ! $   rhipes[i]" Y0, 1 /"{knj](� `  0 $�00 �sa�f.be$c_sh�xe1 =!jh.ceil8np6arr!{*sh�pE�) * amg_shke / Sv2ade ��0!d)aStyp'Int)`" svR)dm
*�� �    # Gache Mma'�Q ift/ zAE�fi�k f/r(fastev 4raklif' )WARNMNG( lar�E"datasa`s`}Ay1dxcued sxwtum rdqkuraes) 0$   "1salF>mes = [nooe]$*�fJ        selv.npYUgk�e3 5 _Pcth(f+$hdl_sufwix*',|py�) for�g2in welfau�f)les]
     !  Iv�casxu[ii�wes:�  � 8� ` p �gb 5,  # Gdgqbytes .f"ciCled �maweq
     `0`0!0 sglfnhm�h&0,!s%lv.imhw�="�NnnE]`
@n, �Oone](.an(  4 "  $  � fcl = selz.caBleWme`gact�_�is+ hf�c�che^iiege[ -� 7dis�'8el{e Smlwloc�_��Age
 " $  �! �( ve�unts(9 uhreatP�{l(nU�ODHRaAF�).i�ap(Fcn.0pcnGe1n))
""0    � $  pb!� = tsdm(%nu-e2aue zesUl6s(, t/t!h?~,�b!2o�gr-!p,BAR_VOROT,@di�a`le=lgCA\_WAnK!>!1i
&`"�!! * &$0For!i,"y on p�er:
        !    d  if c�Cxu�ma�es =9 %fis{&:*   (  "� 0 $�"  ! ` gb += 2eDf*n2y_�I\dw[i].stat(+.wd_size   (  0$$   $ 0$edsi: $� 7rae%
 !  ! "� $   `   *� sEd�mm{mM, {elB.im_lwxiY$�sehf.im]hw[kU  x) # im, `g]�)v,mw_recir\D -0loed_imag�(sefn i)
 � 0  "   �      ! 0gb += s%�gNimsI�&lb{teS     $ "  `1    xbar.deqk�= d'z0reFix<Caa|Ing"ima�g{!(zg� /!ye9:.!&=GB ;cqche_imAga�}ie
   ($   0 Pb`r.close(9

  0 `e"&ca3lo^labeh�(sElf< tatz?Path)/./la"els.!eahe�,, 1zgfi�=''+z
! 1      Sache tetis�t |�bels� cjekk images yjl ru`d(wh�pEw.  0$`�� Xf< {} !# dicuJ    $0  na, of, nuh ~g,0!scs =1 , 0� :, d�![}" # .umRmv(miss[n�l fM}n$, empty<(corrup|, �!sSafes
 ��""4  dmCc =0f�;pr�~i�|c!nnhng /{patj>para,t � pc4H&stem}#`image7 end(lacgls...""    �" wi�H!@gl,MUM_THZEATS) is!AoL:
   a`$     p`aR�=0tqnhqoo$>i}q0)vdsify_imhge_�Abel, zixhqelf.am_fknes rel�,labE._vklgr, fkBeqt+p�efix))�,0  (   0 �  �$   `p0"  (egcc=d}sk, 0           "    ("` �ov!l5lenseiv./m_@k�ms)/�  0`   0  $( $� $� !   �aR_&ormap?BAz_VMW]AT9	   #� !$"``&or ym_fimg,!lb, 3(cp� {e�-nps0lm_�, nff, nm_f, .c_fl"ise in pfar:
  `�$  %     "!(��0/5 n�_f
 ( ("  $ !  0 ( �f +=".f_d      0  !  0 �ne �- ne_b
"!0P   !�0 (   m"@+-�nb_f* �` ( � &  0�  mf im_fhle:
 ! $ 0     $�   $  pZal�fkle]$� [lb< shaqe, segm%ntcU � ( �`       ((mf msg:�$    $ $2      ` !  Msgs/�p�e.d)esg) "0`�1 (   0 d pb�r.d%qc0= z"{desc}{lf} �U��,@9.m= �isqin',�{na} 7mpty* sna�!coRr5rt&  "     pfa2.cngwu(*  ( �   if msow:
   "a d"   �OG�EZ.inbo8'\n'.zo�nhmsG{-)
b0   8 �it:jf ==$0:
0"$       ` hOF��Z*~arkhhG(f'stre�+x}WIRNYNF: _o"d�cE|S!foU.d ij)tatH}.�yHEL_D-')
 ! " �$ h[h�sh/]�= get_hash8sehB.habel^�il�s!+�self.im^fk`ea)
  "a$`` h'smsultr'] -*of, no-(�el nc, lun(sulb>im_filec)
�       pZ�msgs%] = mcGs0 #(girnilgq
`!  2 0 yY'wErwion+] = �elf.cecde[version  � chcaa vurzion
   �  ( pBy:
  #  08 $  nrscve(r`|x((�)( c s`V�$�aclu Fo�neyuxtyme
       "   (�a4h.wad�_suVfAx(/&ccChe.x}%).Pwname(path) �# r%|ove .~py se&bix
          p(LNGGER>inF�('{|r%V�x]ND7cewhe creat�d�`{4aUh}')
  ! $"($excep4"EiCmptcgnd�S�e�`�(( `�   � lGGUV.tarning(f+{ptd&hxWAR�INO: Kac(e dlrectkry {Pqth.pa{ent} is nnt g:kteabn5: {�}')  ' l.|!�R`peAbluJ��   @  ret5rn�x
q  dUn�]_�a.__�QelD):J0 d%    retu�n8leN(Se~g.�m�fimLw)
* 0  ��dmf ?iter]_(seLf)2(   !'(` �"3e|f.couFt = %1 ` 0! �� prin�(grqn(`quawettidew#(
    #�  � #sdlf.�hufnn�d_~actNr�=%np'RanDo�.1crlUtavmo.(senn.oV) hg selva�g]ef� mn3e &p.ir �ge(3elf*~v)*    #   ` beturl se^f�
h   eef ]_getiteI_W(reLf� ineex)>� $$(    ind�h = sa�f.)nD)susSIn�mX]   liNeAr,"rhu�fled os �o�wa�v�ighfs
    " ` Hxp�=`s%lnnhYp�`"    )`mOsaic < vFlf?mo3ai+ a�d!raldom.rin`o}() > �Yp[/mo�aicgU
 $ �d   if e�sa�c:
   0�  "! #�Loa�(mgsaqc* 0         !img$ labels =$self.|o!E_Mosahi(ao�Ez( `&       (rXaqE{ ="No�e�
  "(      " # Ly<Ur0cUwm%nuatim~
   0 �  $ � lf randk-.ranl/m((�~ Hyp['}i(up�~:
    "a `      `(img,$libe~s - m�x�phimg, la�D$S(#
se�f.loAd_losam�(ran$Om.Randilp(/!wgnv.� - 19))

`  ,"` $dlsE*
( `     " ( # �/a$(mmag�* ` 4�! !  " )mg, (h0<�w0i- (x,"w)!= seLv.moad�amage(induy)

     (`     # Lmttebbnp: `        $ share = SeLb>batch_si!pdS[w�|F.savc([m�de�M]1)b �elb.rd�d#�lcdaelf&i�N[si<e   finaL h�tte�bnxed(shape
0(`  (    ��amG,"ratjod`�et0= lgttE2boy�imf, s(pd, a|to}FalsM, scameup=we,d.a=gmgn�)     !0 $  r�aper = )h0(0wt%- ((x/�j0- W0/ g0), bad9  " foR,AOC mAPpe�caling"
$   0  �  0 l�bems"?(self�lqkEhsSiNdex�&coby(+
 ("  �� 0 0af lqb%lssize:  c�lor�alazed$xyu( |o rkX%d X�x{ f�r}ad       "!  $    �a�elS[:*(%:] , Xy�.N2xyXy(l�b!l[[* 1:Y, � 4imK0X(: w, rqTio[qT *�h, qqew-bc$�0M,Apathpa$+1])N
`  !!  !%(  hf$seld.a�gm�nt;
  ( `       $ p �-g, Lkfwlw(; ranD�m_Perspmct}veieg,
     "    !$  (P!  $     b  !�0� �  !  0(     ` |ajeiq$     �  %�0a`   �0      "  `%  0   "  %&� $ �  dee�ees?h}p['duwre%s'M,
 ( $       `(  $   " $ � ! `     $0    ( "h   a  translate=h]p['dra�sdeTe'u,
   (0,`    0 "      !   `� ( 0 "   $ �  0  �!�  ccal�=)y1[osC`lg'Y,
   b  `` $      `$�   $     "&   2      `     8 whear�hy0['sh%d�.],
  ( * $p  ( 0! `"($0    `  ! � b   �!  0( ( "   "pep3pdctiv�=jy�[!pe�sxectivE'],

!  `� &(nm�= lGn lcbGls)  #0Nu-be�,/f la�e|s
$ b$ !" ig n,:
   �(     ( mibels[z((1:5]0="xqxy:�xhn(�ab�ls[:, 1:4M, w=iig.�h1pD3},0hy�f.b`apeZ0],�cli =Vrue( exs=1E-3)

  $" "0 if se,f.a}gm%nt:*`     2     ! Alb�}envetiohs 0�  !    ` imf, la�elr"= sa|f.albumenpa4i/nQ8ing. lAje,z)
(   0"#  (  no$= nen(ma��lsi" # Updae(a�t�2�albumentqtions�
$ D      �  '(HSV(colosmspacu
` �    h  ! a�'oenuhwv�+mg, hwakn/hyp['lsV_h'],`sgci�=jyu{&hrV_s'^. vfain5hyp['hsw_v7])�
    `   4 ()#$Nl,p Upe/wn
(� "   "" � (f$ra.dg�.b!~Dom((h= hyp['&dipuD/]
    �     .!  !iUo -!nX.flittd(maG)
2` #  �  0     d�f n(:& 0   �    �  $    �mq�u�s[8, 7] < s ) ,a"el{[:- 2_
     0  �   +�Blkp mefT-Rmght
40 � )0     )f qa&d�m+raldgi*) < hy�['gliplr?\>
(!� h  ! !$   y/g } npfLi�lR9i�g)
        � 0 � `)v�Jl
    "2     ( �!b   libelcK;-�1]!5  `lqbels[:, !]H  )!$!   � (" CqDoqts `  $  0$0  #l)bel{ = ceUou�(ymg, -ardns� P90v3	1   " $! H #0nl =�,e~(najels) `# UxdkTe dft�r cu$�utj   r �elaje,w_oqV = �orch.zg�os((nl, vi)
 �  d(!0if dl:   `   `�   labgls_guT[d(1:}2=$tOrch.brm^n7mpihhAbgly)

9! !4!  #�COntuxt   (` 0 �mG =(imE/trancpn{e((2, 0, 1)�[:*-1]  # EC t/ BHw, BGR t� RGV
�     ` ikG`=dnp>asco�tkgu�usa2sa��)mga�
 ! �  &$re�u2n |oRcl.f2o�_nUmp�hamo)� lafe|�_oUd= sal.,imbkleSYhje%x_, �(axeq
�( � lmf load[imaGe,sel�. !):
 * , 8 $� Lats 1 image,from%data{et ifdex-ig, Returns`(ii, r�fanal xw {esiXed hw1      ` ie| f- fn < S}lfims�m\,$weMf.imOfildb[a, Rgnb.np=_faldsYi],""  ,$  in!i- }s Ngna: (# nnt b�chdd inRAM( " 0    d cF Nl.exks4�(+:  #noCd!np}  `  � ` �`    �i} = np>eo�d(f�+$ 8�   0  `else8" # rmad ilageZp$      �   � ! iM = �v2.imRgad*v)!#6FG
  (  � d`  �$ ``�{se2T am is jov`Nona, f%	mac� not Ru~d {&}'"    (�   , lp,�g� =�imsjape[�2]( c �ri'bhw
    "(  `` $2 )�sml&/mmE_Size /�mc|(h0, W + '"ratin
!    $"!1( $mf r$%? :  #,iF siz!s mrgl�t eaual   $`   "�� ,   interp`=$c62.HNTEV_LYNeIR0id$(�el&.aqgmml4 or Z 6 1) elqe"�v6,IOTES�REA
       �  ! ) � im } cv2.reSira*qm,"*mn�(v0���r)E int)h0"* c))( �.terpolaTion=in|er`i
 ! 0(� �    r�twr. km, (hp$ w),`me.qhape[82]` ##iml hw�oz-ginal� hw_reSmred
$   � ( revurj sgMd(kmw[i], sedb'�EWhw1�iYl kml�.H)_pw�i]  +<il, �f_origk~al hs_res){�dN    duD g�che^imageq}t_disc(se,g(m): (��  - #`S�veS �n0km`ge as`q, jn*py file nor fes�a�"lnadilg
 `0   " f = self.n�fhle�[-} (�`0 0 ib nOt f.uxirts89:�`      * (  np*sqwe�f.As0OsiX). g�.ims�Ad(salfil_�IlesK)]))�    $e�@�mqd[eorak#hself,"ilfgy):
  `! !203 yLOw 4=e3#ic!lk@e%s. Lo�$s@1 ilege � 3 rC~$�M ima'ew in|o�a!49imaa1koreis "�!  � hsbEl34, segments4 - [], []
! 8 d�  s = s�|f.meg~si~%
� 0�    y�$ xc!} (iNt(rakd�m.1oifobm -�- 34��s + {!+ f�0x hn��el�mocaiC_bo�de2) (� eo�ayc �entgr$x, y    �  "Ind	cds(= [�n$mx] +�bin�m.q(oice� se|d&iN�+cgs, +9)`"# 3 !&di|ion�n iecfp [ndices $ (  $ r`nto�nshufg�e(yn$age�)  !�"  fn`0i  indAx()j!En�-eR`t�hndiaEy)0
`    "    $ %%DoA� im;ge     "0  8 $img, _$ hh, w� @relb.L�ad_imae(yndmX)

  0     �  1# p,abe")lg`in Imf4
 `      !8 $�b a 9= �:  !(pop neft   ` 0!�� !  % 0imo4 = .p&fw\l8(� + 2, s + 2, ymg"�ja�c�2E)l 114�(dxy0e=np*uant8+ `# "ase�I)ige wiup`4 files $ "0  "  $  0 x1a, y1q, 82`, 9�"= m�|(|� -Pw, 8) mih8ya -(H,�09- |c,�xs  !4xmif. 9min, Xeax, =la� 8l�scm kmege)J      0 #p 10   x1b� i1b,�x2b,0y3b =� � .|2a , h1 )- �"} ({:cm0{1g), w, h "#(xoin,hymin, p}�x& y-ix�(sMall ylagm)
 ` * �   `  uli�i �< 1:  3 tg�pzighs�   � !      �  @�1a( y1a, x2e,�y2i = xc, }ax8}# - h, 0),0Oin*yc`+hW, q*(2(. 9
$       0@$�0   �1bf y1`, h2b0}2" = 0 h!)`(q:a`- iqe)((mmf�w,�82a -`x3C)(  
 a !       ��lif(i"<= 2: f#!botuom detv   � �( " � (   z1a y1i, h2e&!y2a =`maxxs�m w, 0), 9c,"|cl$min8s *0, yc�$h)J            �  `Z1b, y1b$ xf�< �2b�5$u -x(x2a  x1a)<"1, w."Min(�"a '�y1`( hi
 0! $ b `)  eLif � =="3:(#��nt</m r�ght
"    `0   "�   (p1a,!q1q, h6e$ y�a 5(\", xc,$mi.(xc ; v,$;8* "�,(miJ(r�**:. yc + h)
   !`   $($a(   y�b, y1�, x�c"{:b = 0, 8, ein(7,h�2a�- xa), m�N(y2a -(y1I, h�
�$"� ��     0iMc[�1�:x2a, |qc8x6q]"= mm�_y5c;u2�( x1b;|2b�  #0k�'4[y}i.jym!x- xmin>qmax�
 &`  �  $  pq�w$=�x1! -!x1" $  ( !0  !!palh25 m1a - y5;H b("2 p�!   #pLab�ls     %  (  (lcJels,(segmeLts = reld.lccels[k\e�p].QnqY(�, selnnseemelts[i��exM6co2y()� �   ( `! #!if La2elB,sizu:
0 !(  "!`(    ! labelq[*� 1>Y } x}whN0h}x9)lakeLa�:,*0�], g� hh p�dw, qaul9 �# Nor�a$Ired x{wh to piXem"xyxy dmRLq|        � "    (s�gmm.us - [xyn xy*x, w$ h, pydw, xaD�)"for$h �n se�mg.�s}
 �         �abuxqtnar`end(lare,s!
 ""     "   qegegntb<.e�ten�)sa'me�6s)*" 0  $ �# Go�caT/clir!h�rens
    ( $�lafenS6-�jP&concAtulat�(Dabels4,(0(#     $(fov!h Yn (larems4K>, �:], �3egmeots<+:�     �""�( !np.cmi�(p, P<2 *8s, ku|={)#`# fdip(wHEj qsing baNdOm_pevwefdiWe )
 ` $ 	  3!img6< mabels4 5�pepl�c�ve�iEg4, ,cbg�e4)  # repl)cate
0  � " 1#`Awg}Gn0 �  ("  kmg4, labmLr( sileots4 =!snpypasTe(img4, l�elw4< seemenVs4,�p<sElf.hxt['�oPy_0esta�M)
"       im�4,0l`bDLs4  rAn�om^xerspect�v�*iMet,
   2" $  "%     d ""! "  !a  � 0    `  $   label{4,
  " `" " " ` "0 )    4    $ ` `  "�b�    seGmEltS4,*@#��0`` !     "    2 !  , p  "(      #  $bd�'z`es9{%ln.hypKdegr�es�],
(`  d" 0 ! �(p    0( 0 !    ( 0 ! 0  �    �anslate-{elf.hy`['uranSlap$'],*! 0 !!1 $ ( !`   `( !(�� � $0  (  1  (  �8 scale=radf.`{t[Gsc�le�]l
   `($      �  !  �((  r  �0(  �! �( ( �  sEaS=smlf&hYp['{hg�r7],
 !�  "(!  (�  ) 8! !  "h !    $   ( b   (``e�sPec4hve=wm|v�hyp['pgrs|mcTi6e'}$
       ! #��    ! 4  0    d `(   "p   � " `bor$er=se`f�Moyiicb/rDer(� c bor5eR }O remoRe
8 "  $  returj �}g4$"maBgL34J%  !lef`loqf�mo#ik�9(sadf,!iftex)z
 � �    !(YOLv$ 9-mosamc loed%r.!Locds 1 ima'g + 8 RAndme$)iages(ifdm !!��imege mosaiC
     $  |1b�ls9,$sec�e~ts8 =p_], [
   4  z = wdlf6�hg_�kje�  `� 0  if�ic%s =�[in�d_0+ rA�Doo.cl�ices,sehn.k~da!e1� k=�)  #!0 alDitio.al imcge`indkses
     `  random�shuf�le(iLdiCes(
` $0   hr(0W` � ,0?�1` # �eiGhv, g+dth pzE6�l�s
  �"   "Fo� y,"inluh iv*elt-esade(iOdic%�):
  " �   0 " b Load i-�'g
 `   �$$    img. _(`,l, w)$=SelN.loeV!Meg�(index	
�    $  $ � #!,se(iMg�af ime9
  a( `�"    if i/==!0>(0# ce~ter
   "     �    � hMO9!=`np.ful�((� *$3,�s " 3,`xmg+qha�m["]�, 114, d41pe=np.uin�)" # base im`ge wid( 4(thDeq
     (a   � !q$h,4s0 5 h�u
0$     ,  $!4  !c ="s< �, � + Ul s`#!h@  �din< ymig, Zmaz`ymax$�b�su) co��dinqtms
     ( �  ` %ni& i?`1 �# top
t `  (0` $ (( ( k 5 s< s - h, g$/ w�$�
0p`( (   �! e]ig i m< 2: �) tOX �h'ht
 0)� #$`` ��   $# = s + wp,s�-0H. s!+ sx � g,$s
 `     $    Umyf!i == 1  # righ䊠  $$("  �     "w =0{d	 w0,*s,�r(+ W0&+rw, s +�i "+        �mliN m-14: (# bmtto� bi��t
 ` "    $ " ` ag =!s"+ w0,s + hp. ��;(w `+0�, W(; Hl!;`(
    "�  (!  emin i ?5 5>" � b/ttOm
�  #""& "` %"    �"3*$v0$m w, s�/ h0(�s + w�, r + h0 +!h*  `� !      ehid i <=`&:  %$ro�dgm left
                c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
            elif i == 7:  # left
                c = s - w, s + h0 - h, s, s + h0
            elif i == 8:  # top left
                c = s - w, s + h0 - hp - h, s, s + h0 - hp

            padx, pady = c[:2]
            x1, y1, x2, y2 = (max(x, 0) for x in c)  # allocate coords

            # Labels
            labels, segments = self.labels[index].copy(), self.segments[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w, h, padx, pady)  # normalized xywh to pixel xyxy format
                segments = [xyn2xy(x, w, h, padx, pady) for x in segments]
            labels9.append(labels)
            segments9.extend(segments)

            # Image
            img9[y1:y2, x1:x2] = img[y1 - pady:, x1 - padx:]  # img9[ymin:ymax, xmin:xmax]
            hp, wp = h, w  # height, width previous

        # Offset
        yc, xc = (int(random.uniform(0, s)) for _ in self.mosaic_border)  # mosaic center x, y
        img9 = img9[yc:yc + 2 * s, xc:xc + 2 * s]

        # Concat/clip labels
        labels9 = np.concatenate(labels9, 0)
        labels9[:, [1, 3]] -= xc
        labels9[:, [2, 4]] -= yc
        c = np.array([xc, yc])  # centers
        segments9 = [x - c for x in segments9]

        for x in (labels9[:, 1:], *segments9):
            np.clip(x, 0, 2 * s, out=x)  # clip when using random_perspective()
        # img9, labels9 = replicate(img9, labels9)  # replicate

        # Augment
        img9, labels9 = random_perspective(img9,
                                           labels9,
                                           segments9,
                                           degrees=self.hyp['degrees'],
                                           translate=self.hyp['translate'],
                                           scale=self.hyp['scale'],
                                           shear=self.hyp['shear'],
                                           perspective=self.hyp['perspective'],
                                           border=self.mosaic_border)  # border to remove

        return img9, labels9

    @staticmethod
    def collate_fn(batch):
        im, label, path, shapes = zip(*batch)  # transposed
        for i, lb in enumerate(label):
            lb[:, 0] = i  # add target image index for build_targets()
        return torch.stack(im, 0), torch.cat(label, 0), path, shapes

    @staticmethod
    def collate_fn4(batch):
        im, label, path, shapes = zip(*batch)  # transposed
        n = len(shapes) // 4
        im4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]

        ho = torch.tensor([[0.0, 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0.0, 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, 0.5, 0.5, 0.5, 0.5]])  # scale
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4
            if random.random() < 0.5:
                im1 = F.interpolate(im[i].unsqueeze(0).float(), scale_factor=2.0, mode='bilinear',
                                    align_corners=False)[0].type(im[i].type())
                lb = label[i]
            else:
                im1 = torch.cat((torch.cat((im[i], im[i + 1]), 1), torch.cat((im[i + 2], im[i + 3]), 1)), 2)
                lb = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            im4.append(im1)
            label4.append(lb)

        for i, lb in enumerate(label4):
            lb[:, 0] = i  # add target image index for build_targets()

        return torch.stack(im4, 0), torch.cat(label4, 0), path4, shapes4


# Ancillary functions --------------------------------------------------------------------------------------------------
def flatten_recursive(path=DATASETS_DIR / 'coco128'):
    # Flatten a recursive directory by bringing all files to top level
    new_path = Path(f'{str(path)}_flat')
    if os.path.exists(new_path):
        shutil.rmtree(new_path)  # delete output folder
    os.makedirs(new_path)  # make new output folder
    for file in tqdm(glob.glob(f'{str(Path(path))}/**/*.*', recursive=True)):
        shutil.copyfile(file, new_path / Path(file).name)


def extract_boxes(path=DATASETS_DIR / 'coco128'):  # from utils.dataloaders import *; extract_boxes()
    # Convert detection dataset into classification dataset, with one directory per class
    path = Path(path)  # images dir
    shutil.rmtree(path / 'classification') if (path / 'classification').is_dir() else None  # remove existing
    files = list(path.rglob('*.*'))
    n = len(files)  # number of files
    for im_file in tqdm(files, total=n):
        if im_file.suffix[1:] in IMG_FORMATS:
            # image
            im = cv2.imread(str(im_file))[..., ::-1]  # BGR to RGB
            h, w = im.shape[:2]

            # labels
            lb_file = Path(img2label_paths([str(im_file)])[0])
            if Path(lb_file).exists():
                with open(lb_file) as f:
                    lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)  # labels

                for j, x in enumerate(lb):
                    c = int(x[0])  # class
                    f = (path / 'classifier') / f'{c}' / f'{path.stem}_{im_file.stem}_{j}.jpg'  # new filename
                    if not f.parent.is_dir():
                        f.parent.mkdir(parents=True)

                    b = x[1:] * [w, h, w, h]  # box
                    # b[2:] = b[2:].max()  # rectangle to square
                    b[2:] = b[2:] * 1.2 + 3  # pad
                    b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(int)

                    b[[0, 2]] = np.clip(b[[0, 2]], 0, w)  # clip boxes outside of image
                    b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
                    assert cv2.imwrite(str(f), im[b[1]:b[3], b[0]:b[2]]), f'box failure in {f}'


def autosplit(path=DATASETS_DIR / 'coco128/images', weights=(0.9, 0.1, 0.0), annotated_only=False):
    """ Autosplit a dataset into train/val/test splits and save path/autosplit_*.txt files
    Usage: from utils.dataloaders import *; autosplit()
    Arguments
        path:            Path to images directory
        weights:         Train, val, test weights (list, tuple)
        annotated_only:  Only use images with an annotated txt file
    """
    path = Path(path)  # images dir
    files = sorted(x for x in path.rglob('*.*') if x.suffix[1:].lower() in IMG_FORMATS)  # image files only
    n = len(files)  # number of files
    random.seed(0)  # for reproducibility
    indices = random.choices([0, 1, 2], weights=weights, k=n)  # assign each image to a split

    txt = ['autosplit_train.txt', 'autosplit_val.txt', 'autosplit_test.txt']  # 3 txt files
    for x in txt:
        if (path.parent / x).exists():
            (path.parent / x).unlink()  # remove existing

    print(f'Autosplitting images from {path}' + ', using *.txt labeled images only' * annotated_only)
    for i, img in tqdm(zip(indices, files), total=n):
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists():  # check label
            with open(path.parent / txt[i], 'a') as f:
                f.write(f'./{img.relative_to(path.parent).as_posix()}' + '\n')  # add image to txt file


def verify_image_label(args):
    # Verify one image-label pair
    im_file, lb_file, prefix = args
    nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, '', []  # number (missing, found, empty, corrupt), message, segments
    try:
        # verify images
        im = Image.open(im_file)
        im.verify()  # PIL verify
        shape = exif_size(im)  # image size
        assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
        assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'
        if im.format.lower() in ('jpg', 'jpeg'):
            with open(im_file, 'rb') as f:
                f.seek(-2, 2)
                if f.read() != b'\xff\xd9':  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, 'JPEG', subsampling=0, quality=100)
                    msg = f'{prefix}WARNING: {im_file}: corrupt JPEG restored and saved'

        # verify labels
        if os.path.isfile(lb_file):
            nf = 1  # label found
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any(len(x) > 6 for x in lb):  # is segment
                    classes = np.array([x[0] for x in lb], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
                    lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                lb = np.array(lb, dtype=np.float32)
            nl = len(lb)
            if nl:
                assert lb.shape[1] == 5, f'labels require 5 columns, {lb.shape[1]} columns detected'
                assert (lb >= 0).all(), f'negative label values {lb[lb < 0]}'
                assert (lb[:, 1:] <= 1).all(), f'non-normalized or out of bounds coordinates {lb[:, 1:][lb[:, 1:] > 1]}'
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:  # duplicate row check
                    lb = lb[i]  # remove duplicates
                    if segments:
                        segments = [segments[x] for x in i]
                    msg = f'{prefix}WARNING: {im_file}: {nl - len(i)} duplicate labels removed'
            else:
                ne = 1  # label empty
                lb = np.zeros((0, 5), dtype=np.float32)
        else:
            nm = 1  # label missing
            lb = np.zeros((0, 5), dtype=np.float32)
        return im_file, lb, shape, segments, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f'{prefix}WARNING: {im_file}: ignoring corrupt image/label: {e}'
        return [None, None, None, None, nm, nf, ne, nc, msg]


class HUBDatasetStats():
    """ Return dataset statistics dictionary with images and instances counts per split per class
    To run in parent directory: export PYTHONPATH="$PWD/yolov5"
    Usage1: from utils.dataloaders import *; HUBDatasetStats('coco128.yaml', autodownload=True)
    Usage2: from utils.dataloaders import *; HUBDatasetStats('path/to/coco128_with_yaml.zip')
    Arguments
        path:           Path to data.yaml or data.zip (with data.yaml inside data.zip)
        autodownload:   Attempt to download dataset if not found locally
    """

    def __init__(self, path='coco128.yaml', autodownload=False):
        # Initialize class
        zipped, data_dir, yaml_path = self._unzip(Path(path))
        try:
            with open(check_yaml(yaml_path), errors='ignore') as f:
                data = yaml.safe_load(f)  # data dict
                if zipped:
                    data['path'] = data_dir
        except Exception as e:
            raise Exception("error/HUB/dataset_stats/yaml_load") from e

        check_dataset(data, autodownload)  # download dataset if missing
        self.hub_dir = Path(data['path'] + '-hub')
        self.im_dir = self.hub_dir / 'images'
        self.im_dir.mkdir(parents=True, exist_ok=True)  # makes /images
        self.stats = {'nc': data['nc'], 'names': list(data['names'].values())}  # statistics dictionary
        self.data = data

    @staticmethod
    def _find_yaml(dir):
        # Return data.yaml file
        files = list(dir.glob('*.yaml')) or list(dir.rglob('*.yaml'))  # try root level first and then recursive
        assert files, f'No *.yaml file found in {dir}'
        if len(files) > 1:
            files = [f for f in files if f.stem == dir.stem]  # prefer *.yaml files that match dir name
            assert files, f'Multiple *.yaml files found in {dir}, only 1 *.yaml file allowed'
        assert len(files) == 1, f'Multiple *.yaml files found: {files}, only 1 *.yaml file allowed in {dir}'
        return files[0]

    def _unzip(self, path):
        # Unzip data.zip
        if not str(path).endswith('.zip'):  # path is data.yaml
            return False, None, path
        assert Path(path).is_file(), f'Error unzipping {path}, file not found'
        ZipFile(path).extractall(path=path.parent)  # unzip
        dir = path.with_suffix('')  # dataset directory == zip name
        assert dir.is_dir(), f'Error unzipping {path}, {dir} not found. path/to/abc.zip MUST unzip to path/to/abc/'
        return True, str(dir), self._find_yaml(dir)  # zipped, data_dir, yaml_path

    def _hub_ops(self, f, max_dim=1920):
        # HUB ops for 1 image 'f': resize and save at reduced quality in /dataset-hub for web/app viewing
        f_new = self.im_dir / Path(f).name  # dataset-hub image filename
        try:  # use PIL
            im = Image.open(f)
            r = max_dim / max(im.height, im.width)  # ratio
            if r < 1.0:  # image too large
                im = im.resize((int(im.width * r), int(im.height * r)))
            im.save(f_new, 'JPEG', quality=50, optimize=True)  # save
        except Exception as e:  # use OpenCV
            print(f'WARNING: HUB ops PIL failure {f}: {e}')
            im = cv2.imread(f)
            im_height, im_width = im.shape[:2]
            r = max_dim / max(im_height, im_width)  # ratio
            if r < 1.0:  # image too large
                im = cv2.resize(im, (int(im_width * r), int(im_height * r)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(f_new), im)

    def get_json(self, save=False, verbose=False):
        # Return dataset JSON for Ultralytics HUB
        def _round(labels):
            # Update labels to integer class and 6 decimal place floats
            return [[int(c), *(round(x, 4) for x in points)] for c, *points in labels]

        for split in 'train', 'val', 'test':
            if self.data.get(split) is None:
                self.stats[split] = None  # i.e. no test set
                continue
            dataset = LoadImagesAndLabels(self.data[split])  # load dataset
            x = np.array([
                np.bincount(label[:, 0].astype(int), minlength=self.data['nc'])
                for label in tqdm(dataset.labels, total=dataset.n, desc='Statistics')])  # shape(128x80)
            self.stats[split] = {
                'instance_stats': {
                    'total': int(x.sum()),
                    'per_class': x.sum(0).tolist()},
                'image_stats': {
                    'total': dataset.n,
                    'unlabelled': int(np.all(x == 0, 1).sum()),
                    'per_class': (x > 0).sum(0).tolist()},
                'labels': [{
                    str(Path(k).name): _round(v.tolist())} for k, v in zip(dataset.im_files, dataset.labels)]}

        # Save, print and return
        if save:
            stats_path = self.hub_dir / 'stats.json'
            print(f'Saving {stats_path.resolve()}...')
            with open(stats_path, 'w') as f:
                json.dump(self.stats, f)  # save stats.json
        if verbose:
            print(json.dumps(self.stats, indent=2, sort_keys=False))
        return self.stats

    def process_images(self):
        # Compress images for Ultralytics HUB
        for split in 'train', 'val', 'test':
            if self.data.get(split) is None:
                continue
            dataset = LoadImagesAndLabels(self.data[split])  # load dataset
            desc = f'{split} images'
            for _ in tqdm(ThreadPool(NUM_THREADS).imap(self._hub_ops, dataset.im_files), total=dataset.n, desc=desc):
                pass
        print(f'Done. All images saved to {self.im_dir}')
        return self.im_dir


# Classification dataloaders -------------------------------------------------------------------------------------------
class ClassificationDataset(torchvision.datasets.ImageFolder):
    """
    YOLOv5 Classification Dataset.
    Arguments
        root:  Dataset path
        transform:  torchvision transforms, used by default
        album_transform: Albumentations transforms, used if installed
    """

    def __init__(self, root, augment, imgsz, cache=False):
        super().__init__(root=root)
        self.torch_transforms = classify_transforms(imgsz)
        self.album_transforms = classify_albumentations(augment, imgsz) if augment else None
        self.cache_ram = cache is True or cache == 'ram'
        self.cache_disk = cache == 'disk'
        self.samples = [list(x) + [Path(x[0]).with_suffix('.npy'), None] for x in self.samples]  # file, index, npy, im

    def __getitem__(self, i):
        f, j, fn, im = self.samples[i]  # filename, index, filename.with_suffix('.npy'), image
        if self.cache_ram and im is None:
            im = self.samples[i][3] = cv2.imread(f)
        elif self.cache_disk:
            if not fn.exists():  # load npy
                np.save(fn.as_posix(), cv2.imread(f))
            im = np.load(fn)
        else:  # read image
            im = cv2.imread(f)  # BGR
        if self.album_transforms:
            sample = self.album_transforms(image=cv2.cvtColor(im, cv2.COLOR_BGR2RGB))["image"]
        else:
            sample = self.torch_transforms(im)
        return sample, j


def create_classification_dataloader(path,
                                     imgsz=224,
                                     batch_size=16,
                                     augment=True,
                                     cache=False,
                                     rank=-1,
                                     workers=8,
                                     shuffle=True):
    # Returns Dataloader object to be used with YOLOv5 Classifier
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = ClassificationDataset(root=path, imgsz=imgsz, augment=augment, cache=cache)
    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    generator = torch.Generator()
    generator.manual_seed(0)
    return InfiniteDataLoader(dataset,
                              batch_size=batch_size,
                              shuffle=shuffle and sampler is None,
                              num_workers=nw,
                              sampler=sampler,
                              pin_memory=PIN_MEMORY,
                              worker_init_fn=seed_worker,
                              generator=generator)  # or DataLoader(persistent_workers=True)
