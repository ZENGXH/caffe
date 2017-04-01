#rsync zengxiaohui@10.10.10.202:/data2/zengxiaohui/experiment/ssn_crop/08*/log.train log/tune_from_1500.log
#rsync zengxiaohui@10.10.10.202:/data2/zengxiaohui/experiment/ssn_crop/12*/log.train log/tune_from_init_limit.log
#rsync zengxiaohui@10.10.10.202:/data2/zengxiaohui/experiment/ssn_crop/06_tsn1_rgb_square_thre_sample_tune_from_ori/log.train log/tune_from_ori_cont.log

## flowheatmap_sto.log train ##
#rsync zengxiaohui@10.10.10.202:/data2/zengxiaohui/experiment/ssn_crop/02_1_tsn1_stosampling_flowheatmap/log.train log/flowheatmap_sto.log
#rsync zengxiaohui@10.10.10.202:/data2/zengxiaohui/experiment/ssn_crop/14_1_tsn1_rgb_square_thre_sample_scaleratio_modi_fix_flip/log.train log/thre_sample.log
#rsync zengxiaohui@10.10.10.202:/data2/zengxiaohui/experiment/ssn_crop/02_1_tsn2_more1000/log.train log/gtheatmap_sto.log
rsync zengxiaohui@10.10.10.202:/data2/zengxiaohui/experiment/ssn_crop/tsn_square_rgb_retrain/log.train log/train_ssn1_from_scarch.log


## echo "compare tune from ori with original way and with sto sample from fix crop -> ssn1"
# rsync zengxiaohui@10.10.10.202:/data2/zengxiaohui/experiment/ssn_crop/13_2_CONTROL_tsn1_rgb_square_ran_sample_tune_from_ori/log.train log/from_init_fix_crop.log
# rsync zengxiaohui@10.10.10.202:/data2/zengxiaohui/experiment/ssn_crop/13_2_tsn1_rgb_square_sto_sample_tune_from_ori/log.train log/from_init_fix_crop_stocenter.log
##

# rsync zengxiaohui@10.10.10.202:/data2/zengxiaohui/experiment/ssn_crop/14*/log.train log/202_13.log
#rsync zengxiaohui@10.10.10.202:/data2/zengxiaohui/experiment/ssn_crop/13_1*/log.train log/13_1.log

echo "conv conv rgb"
rsync zengxiaohui@10.10.10.201:/data2/zengxiaohui/experiment/tsn3_temporal/10f_rgb_tsn3_conv_inception5b-share-crop/log.train log/10f_rgb_tsn3_conv_inception5b-share-crop.log 
rsync zengxiaohui@10.10.10.201:/data2/zengxiaohui/experiment/tsn3_temporal/10f_rgb_tsn3_conv_inception5b-not-share-crop-flip-every-seg/log.train log/10f_rgb_tsn3_conv_inception5b-not-share-crop-flip-every-seg.log 
rsync zengxiaohui@10.10.10.201:/data2/zengxiaohui/experiment/tsn3_temporal/10_rgb_tsn3_conv_inception5b/log.train log/10_rgb_tsn3_conv_inception5b.log 
rsync zengxiaohui@10.10.10.201:/data2/zengxiaohui/experiment/tsn3_temporal/08f_rgb_tsn3_conv3_fix_boxflip-keep-seg-flip/log.train \
    log/08f_rgb_tsn3_conv3_fix_boxflip-keep-seg-flip.log 

rsync zengxiaohui@10.10.10.201:/data2/zengxiaohui/experiment/tsn3_temporal/08f_rgb_tsn3_conv3_fix_boxflip/log.train log/08f_rgb_tsn3_conv3_fix_boxflip.log

rsync zengxiaohui@10.10.10.139:/data1/zengxiaohui/experiment/tsn3_temporal/11_rgb_tsn3_conv_inception5b_773/log.train log/11_rgb_tsn3_conv_inception5b_773.log

python plot_training_log.py 6 \
    tsn3_rgb_conv_conv.png \
 log/10f_rgb_tsn3_conv_inception5b-share-crop.log \
log/08f_rgb_tsn3_conv3_fix_boxflip-keep-seg-flip.log \
 log/08f_rgb_tsn3_conv3_fix_boxflip.log \
log/10_rgb_tsn3_conv_inception5b.log \
log/11_rgb_tsn3_conv_inception5b_773.log \
log/10f_rgb_tsn3_conv_inception5b-not-share-crop-flip-every-seg.log
#    tsn1_rgb_square.png \
#    log/from_init_fix_crop_stocenter.log \
#    log/from_init_fix_crop.log \

    
    #    log/train_ssn1_from_scarch.log \
#    log/13_1.log        

## flowheatmap_sto.log ##
#    log/thre_sample.log \
#    log/gtheatmap_sto.log \
#    log/flowheatmap_sto.log 


#    train_original.log \
#    log/tune_from_init_limit.log \
#    log/202_13.log \
#    log/tune_from_ori_stage1.log \
#    log/tune_from_ori_cont.log \
#    log/202_14.log 

#    log/repeat_iter1700.log \
#    finetune1700.log \
#    log/retrain.log \
#    log/lre3.log log/lre4.log log/lre3_cont.log 
#    train_original.log \
#    finetune3500.log \
#    log/max_center.log \
#    log/repeat_iter1700.log \
#    log/tune_from_1500.log \
#     log/check_finetune.log

