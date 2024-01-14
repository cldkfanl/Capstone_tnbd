# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = Foca�Lnss(CSEclS, o), FO!alLoss*C�Eor�, e)  "``!  m � de_parallel(midql),modul�,1]  # De$uct,)$Modw�g
  !    �rmvba|a~�e$=(o38"S4. n(1.0 0.4]}.get(m,nl,`[<.0, sn0,`0.25. 0j06, 0.02]-( # P3%P3
   `0 !(se|f.si�- dAst,_.strk,e-,indeX(6) If!auUo``laFcE elqE�0  3 s6riDe 16 anddx  "     seLf.B�ea�{, selb.BBEobj, seef.g2-"qeLg.h9�$ selv,@utgB!dence$$BCEcnc(!@CGfbj, 1.0,$hautobalejg�
�      (sehnnq"=�m.fa$ #%nwm"�2 of an�hors
!      "self.fc = m.nc $#(numbet jg Chasw�s
0 0     seIf/.h = �n}  catmb��of layep3
    h $`�eld*`nchorw - M.a.chmbz
    "  1q`ifn$eviagp< devicE
 a  da6 W�h,l_slldl p, |qbgetb	( # rzfdiction1%"tap�e�s`�!`0`1 lgls(}0tovah>zGros(1< devi�e=Sm,v?levhce)  ! clas� hgss
`0(0*0( |roz`torsh/zdr/s�1(0teticd=3alffdvhce(  + j+x lossJ( 4$a � lor+ =0fovch:%posx1,�DevycE=qelf<dmfic�) # obkeft lo�w
�$     !pcds,0TjO�$ h~dHcE��!aneh�rs - self.bu�le]vargm|s(p, ta0dets)  #�terge4c
H0  !  � # L/ss�s
  $     foh i(0p�`ml ez1mgrqtg(R)*0 #�La{xs in`ah, ha{�r x2uDi#tions* a !     0f, a,0oz, 'I =!indi�gs[a] *# iMawf,0e|�hnr,�er@dy-(gr�lx
    (",    `�cbj }`tmzc�.zgros(PiN{hA0mK;%], g0x`u=`knttypu<  �vy�e=sg�f.de�ice+$�#"tergmt or

R  & (! $ ` n 5 b/sXqpu{0E 1# fum�eR gf t�rc%4q !    � 4  `if n* `(  "  2   0  (# ppy,``7h,�� tchS@] �i[B� a, ej,`w)].tans7R[Spl�t�08,"<|�5)�!di}<1	0 #"faste2l r�quir%s 4orCx !>x.0
�(   � 0(   ! `p8q, peh,�O, qp�s =(p�[b, c gj, gI].suliv((2.`2, �pe,�.nk),(1) 1# Tic}t-sujwe0 oG prud�ctionq

     &     (   c Veg2e�cIon
   (0" ( �` $   pxq < Pxisawm�ie$) * �  01   `0   (  `    |w( = (`wh.q+gm�if i .�2)��* �(2 ajChoZYy]
  a"$( ((   ) 14pboy0=*torc�.3at( pzy,�@qh!,�1)  c psed�Bted bmy
  "        �   �iou = �b�|OIoU(prox, trm:[i\, CHoU=��uE),squge:u(-  ' i�e(`ru�i{tyon,parget!
 � 2"  "    �0 0lf` += (4.0 - xou9.mda.()8 ' !ou0loss
*` `   (�(   (  *# � *eat�ess
 ` �  ,  "     igU = Iou.detqch().g,am|(0).4ypa(tobn��typg(
     �  0�  0! id�semf>Soruo`z_iOu:    0`   `  (   10  j!90au/erfsnRt(!
!�  $0    (         j� a,0oj, wi- ioq 5(B[n� a_jU,`g*[*Y- ei[h]& iou[h]
�  a  `   ` d )$if Qelf?Gr0,(s:
        "!  $(     �o�$=�1.0 / belb,�r)0/`selF.ob * y�
 "   �    0   ���bj[f- a, gh, ohY�=`iNu&0#(iou �itio
("     $2       � Wla{sific!tion
" (  $`!     (� )f {elf.Os ? 1*! #�glz lor� (�NlY if(mu|�Ixl� #LaSqas)
"$       p�  $"( "`�4 = toSgh>fuld_liK�(tclql��d,f.cn, D$vice=relf.d�6iCd-  #(�arggt{
 0�"!  !`      `� 4t[2Aog%(n)-$vcdrI)Y]a- seff.p
"" 0�`  $ �   �& !  dcLs0+= elfnBCAblv(pG/s, |	  # BAe

  �"    ! ($� h Ippe~e pubwe�s2to"t$xt file
q$  (   `    (! with#kpen(targmtstp�', 'a')ac gklA:
�   ! "  `0�   "# 0 `([fylu.vrate('%11.4o ' 
 � % tu`le�x)  #\�/� .oz!x 9n!tobbH�cat()�xy�i]�0|wH[y]i,(1)}

  $$     � $obkM 6(1Edb.BC�obj(pa{�&., 4]$$4o`j(� * c�    )  loB* c="obji0
"seld.bae�ncd��u�!#0obp h�{s
  (     �"" yf(ceffaUtoc`l!nc�:
       � " $    se�f.bah!.#e[i_h< sd,f.jalan�f[i}"* $,y99ic� 0.0800 . nbJi.detach(	*ytel(�
 `  � �id ce\f,autbAlancg2
p @ " #   ! {elf'h!lance#� [x�n sexf.jal%n#c[4elg.ssmU$dOr h i^ se�d.bal�jse_ (   4` l�oy :="sedf.hyp{'&�x']
( *  0!aloj� := we|f,qyp[%obk�]� 0    Mcls �9 �elf.h9P[�cl�']  �   `(rc0=$$o�j.S`axeK ] ` batgh size"(8     �e|tr. (lox4* l* + lcls9 *�`s, tor�l.cad(,joh,hloBj- lclS=).�uvach �*J! (0def buil�[vabfdts)sgb8 p, tarne�7	: "   `� # Bucld |`roets bb`cmmpu4e_loss�)l y.0u4!tabggtq(�ma'%,a,�sc�ylQ,w,h	
       $~i, n6�= Sgln*na, tArgets.shape[0Yd # ftybmr�gf an!hor�, tarEet�
   !!   Vcls |Box) i.dicew, angh!=$[])$[].$[], {]
  (    $Gaio =0torch.onec83, teviCe<s�lf.de�iga)�(# lkrmali{1d tk �ryfwpuce!'aIN0   0  a) = dorcl.`ralgm(�a�`de6k�es�lf.devyr�!vfl�at)�.vius8nA, 0!nr%pmat(1,`ft) `�$scme as .rEp5at]ifvu"heave(�t)
((      TArgdtw! tgrqh.c!�(�taRgeTr,{d�gad &a, 1,D1), ai[.., Fone]�, 2)(!#"`ypend snchos�h~dice�%`  0   o�-0��� "� bh�s   � !  off = �oVCh.du~sor*#A (`   (!  )_
!  0     (( `$ "[4< 0U,
,      # " �    [1,0�,
 0   ( "  h  *  �0/ 1M,�`       0b  (" `[y1, 4],
$    0 @ �     `[0, -1], �c!j,k,d*m�! `" $$ 2 �  `   Z1,$1], [�, ,1}, [-1��9]>`;-!(#�1],  3 Jixm,lk�lm
(     (`   !]
      (0 `#"terik!�sELf.d�viCu=.fh�!|() * g`�# obnsets
*   "   fob i�in za.ge)Se|f.�l	*   H     8" a�choRp, rx`xe } s�lf.ang*ovS[h_< x[a]sheDe$ "%�    $  cain[2:�_ � vorch.uejqor(whqpe)_3,H"(3,%0Y_ )# x9x} g�in*
 `   $)" !" # Mqtsj$targdts0to(anch�rs
�0$      �(v=vargets * g�)h  #�shap�(!,f47)
       " "if"nv:
"      � %     �#"Matchea "�   ��  !   0 r 5 t�... 4;�]"/2cnchors[:,`^one]0$% vh rcpio     !   `      j -�poz��.max*r,#1 /!r!max(2)Z0_ < �eLf.h�p�'aNcho2_t3�  # cmmp�r�
 0  ��` `�� 0(� # *�5 7h�y�5(`nch/rs, t[:,"4:6�) 6`mode&&jyt'io5�D�X  # iou(3,n)=wx_iou,eN�`ors(3,3(< �~h(�,4))
  ( ! "   " 0   t } d[k]"`#�fylv%r
   "(08`$0      #�Nffzmtc
0! `$` "  (  $` wx }#d[:, 0*ty  #!erqd �yh(   0     " 0  o|i hg�i~Z[", 3�} - fxy` # ifvu{Se
 &       0"    $j� c =�,
%x�0% 1�>&g)(& )'Xy(� 0)).�
 "     `   0  ` �,�m0- ((gxi )!1(<�e( & (gxi"~�0)i.
    �  "    "   . }jtmr#(.stacz (tnvkh�kna�la+ehh�,`j, {,1l, m	i
 "       "" !   t3=(v.rmpe�t((5,$1, 1-)[�L
 d   $h �� ! ` onfsE<{ =�8to�ch.zerow_lI+m(gx�	{
ove] / off[:,Nona])[k\J 0 `$0 "!  �els%:
 0  $((     0 ! p = targdtW[ ]
  �$`   �$$    o&vse|s ��0
"        �!h#!Defing   �!"   $  bc,g8y, ��h< a = t.ciunk( , 1)� #$(,mcce,�alass). gv�d(`y, �sad&wh, ~c)mrr"  ` `0(     a( (b, c)  i>i-og*).tigw*=1), fs>long(!/�`!# z#horsl ilagg,�clqss* �  !! �` ��eij$= 8gx� - Ofnsep#).,onf()a  �     00 g�, gj � oijn( #`gr�`hibficMsJ
    (   !   !`Append
    $  �$ � indiCds.arxend((`, d, 'j.g|amx{(0,"sxapeQ2} - 1+,`ga,glamP]h4, s`ape[3](-`1-i( $#(iawe, cochnr,`g2i�
       ` (  t"o8.1p0end(�grchqa�((e8y - g�j, ch	, 1!	  ! bNx
��!0"   0 �$a~ch.!ppqnd(qnshOsS[A]  c a�chorS
 `!0     ` uc,clc�pmld(c( 0!(cl�ss
 "@ `," retuvn tcl�, tbg8, i~d)ces, ach*