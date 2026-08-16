function export_matlab_reference(mat_path, out_dir)
% Export the parts of an RA-processed_<mouse>_<date>.mat file needed to
% numerically verify the Python port against real MATLAB output, in formats
% scipy.io.loadmat/pandas can read directly (the trialTable field itself is
% an MCOS/table object scipy can't parse -- must go through real MATLAB).
%
% Usage (headless): matlab -batch "export_matlab_reference('processed_X.mat', 'out_dir')"
    d = load(mat_path);
    fn = fieldnames(d);
    p = d.(fn{1});
    tt = p.trialTable;

    writetable(tt, fullfile(out_dir, 'trialTable.csv'));

    params_out = struct();
    for f = {'ptsKeep_before', 'ptsKeep_after', 'finalSampleFreq', 'finalTimeStep', ...
             'timeShift', 'signalDetrendWindow', 'detrendWindowTime', 'finalSamples'}
        if isfield(p.params, f{1})
            params_out.(f{1}) = p.params.(f{1});
        end
    end
    save(fullfile(out_dir, 'params_flat.mat'), '-struct', 'params_out', '-v7');

    % processed.signals: the double-z-scored trace (pSignal) everything downstream uses.
    sig = p.signals;
    if iscell(sig)
        sig1 = sig{1};
    else
        sig1 = sig(1, :);
    end
    save(fullfile(out_dir, 'signals_ch1.mat'), 'sig1', '-v7');

    fprintf('Exported trialTable (%d rows) + params + signals to %s\n', height(tt), out_dir);
end
