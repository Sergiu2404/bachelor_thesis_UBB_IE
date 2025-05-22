import 'package:flutter/material.dart';

class StockCard extends StatelessWidget {
  final String companyName;
  final String symbol;
  final double price;

  const StockCard({
    Key? key,
    required this.companyName,
    required this.symbol,
    required this.price,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      elevation: 5,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
      ),
      child: Padding(
        padding: const EdgeInsets.all(12.0),
        child: ListTile(
          contentPadding: EdgeInsets.zero, // Removes default padding
          // title: Row(
          //   crossAxisAlignment: CrossAxisAlignment.start,
          //   children: [
          //     Text(
          //       companyName,
          //       style: const TextStyle(
          //         fontWeight: FontWeight.bold,
          //         fontSize: 16,
          //       ),
          //       overflow: TextOverflow.ellipsis,
          //     ),
          //     const SizedBox(width: 8),
          //     Flexible(
          //       child: Text(
          //         symbol,
          //         style: Theme.of(context).textTheme.bodyMedium,
          //         overflow: TextOverflow.ellipsis,
          //       ),
          //     ),
          //   ],
          // ),
          title: Row(
            children: [
              Expanded(
                flex: 2,
                child: Text(
                  companyName,
                  style: const TextStyle(
                    fontWeight: FontWeight.bold,
                    fontSize: 16,
                  ),
                  overflow: TextOverflow.ellipsis,
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                flex: 1,
                child: Text(
                  symbol,
                  style: Theme.of(context).textTheme.bodyMedium,
                  overflow: TextOverflow.ellipsis,
                ),
              ),
            ],
          ),

          trailing: Text(
            '\$${price.toStringAsFixed(2)}',
            style: Theme.of(context).textTheme.bodyLarge?.copyWith(
              color: Colors.green,
              fontWeight: FontWeight.bold,
            ),
          ),
        ),
      ),
    );
  }
}
