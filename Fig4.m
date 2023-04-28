for num=[1,2,3,4]
    load(['test\table1_case',num2str(num),'\result.mat'], 'denoised_result')
    load('data\simu_indian.mat')
    denoised_result = denoised_result([2,3,4,5,6,7,8]);
    p=224;
    metric_curve = cell(2,length(denoised_result));
    for ii=1:length(denoised_result)
        psnrvector=zeros(1,p);
        ssimvector=zeros(1,p);
        imagery1 = denoised_result{ii};
        imagery2 = simu_indian;
        for i=1:1:p
            J=255*imagery1(:,:,i);
            I=255*imagery2(:,:,i);
            psnrvector(i)=PSNR_c(J,I,M,N);
            ssimvector(i)=ssim_ZhouWang(J,I);
        end
        metric_curve{1,ii} = psnrvector;
        metric_curve{2,ii} = ssimvector;
    end

    figure('Position', [100,100,800,250])

    subplot(1,2,1),hold on
    for ii=1:length(metric_curve)
        plot(metric_curve{1,ii},'LineWidth',1.5)
    end
    xlabel('Band Number')
    ylabel('PSNR')
    xlim([1,224])

    subplot(1,2,2),hold on
    for ii=1:length(metric_curve)
        plot(metric_curve{2,ii},'LineWidth',1.5)
    end
    xlabel('Band Number')
    ylabel('SSIM')
    xlim([1,224])
    if num==5
        legend({'BM4D','RPCA','TDL','LRTV','LRTDTV','LMHTV','LTHTV'},'Location','best','Orientation','horizontal','NumColumns',2)
    end
    saveas(gcf,['figs\Fig4_case',num2str(num),'_metric_curve.eps'],'epsc')
    saveas(gcf,['figs\Fig4_case',num2str(num),'_metric_curve.jpg'],'jpg')
end