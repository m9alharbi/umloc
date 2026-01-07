ade = metric_fn['ade'](pred_traj_fake, gt_traj_real).mean()
                fde = metric_fn['fde'](pred_traj_fake, gt_traj_real).mean()
                

            ade_outer.append(min_ade)
            fde_outer.append(best_fde)

            ate, rte = compute_ate_rte(pred_traj, gt_traj_real, pred_per_min)
            pos_losses = np.mean((gt_traj_real - pred_traj) ** 2, axis=0)
            pos_cum_error = np.linalg.norm(pred_traj - gt_traj_real, axis=1)
            error = np.sort(pos_cum_error)
            cumulative_prob = np.arange(1, len(error) + 1) / len(error)
            log_line = format_string(data, np.mean(pos_losses), ate, rte, min_ade.item(), best_fde.item())
            if log_file is not None:
                with open(log_file, 'a') as f:
                    log_line += '\n'
                    f.write(log_line)

            ate_all.append(ate)
            rte_all.append(rte)
            print('Sequence {}, ATE: {}, RTE: {}, ADE: {}, FDE: {}'.format(data, ate, rte, min_ade, best_fde))

            # Assuming the first column is 'x' and the second column is 'y' for both preds and gt
            time = np.linspace(0, len(pred_traj)//fs, len(pred_traj))

            plot_result(pred_traj, gt_traj_real, error, cumulative_prob, pos_cum_error, data, args)
    ate_all = np.array(ate_all)
    rte_all = np.array(rte_all)
    ade_outer = np.array(ade_outer)
    fde_outer = np.array(fde_outer)

    measure = format_string('ATE', 'RTE', 'ADE', 'FDE', sep='\t')
    values = format_string(np.mean(ate_all), np.mean(rte_all), np.mean(ade_outer), np.mean(fde_outer), sep='\t')
    print(measure, '\n', values)

    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(measure + '\n')
            f.write(values)